"""
MADDPG 算法的 PyTorch 实现，与 maddpg.py 一一对应。
"""
import numpy as np
import torch
import torch.nn as nn
from maddpg.common.distributions_torch import DiagGaussianPdType, make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer_torch import ReplayBuffer

try:
    from gymnasium import spaces as gym_spaces
except ImportError:
    from gym import spaces as gym_spaces

POLYAK = 1.0 - 1e-2
GRAD_NORM_CLIP = 0.5


def bellman_target(rewards, dones, next_q, gamma):
    """One-step target used by the official trainer."""

    return rewards + gamma * (1.0 - dones) * next_q


def actor_loss(q_values, flat_policy_parameters, regularization=1e-3):
    """Return total actor loss, policy-gradient term, and TF-style p_reg."""

    policy_gradient_loss = -torch.mean(q_values)
    policy_regularization = torch.mean(flat_policy_parameters.square())
    total = policy_gradient_loss + regularization * policy_regularization
    return total, policy_gradient_loss, policy_regularization


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1.0 - done)
        discounted.append(r)
    return discounted[::-1]


class MLP(nn.Module):
    """MLP：两层 ReLU 隐藏层 + 线性输出，对应原 tf layers.fully_connected。"""

    def __init__(self, input_dim, num_outputs, num_units=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_units)
        self.fc2 = nn.Linear(num_units, num_units)
        self.fc3 = nn.Linear(num_units, num_outputs)
        self.reset_parameters()

    def reset_parameters(self):
        # tf.contrib.layers.fully_connected defaults to Xavier weights and zero bias.
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def clip_grad_norm_per_parameter(parameters, max_norm):
    """Apply TF1 ``clip_by_norm`` semantics independently to each tensor."""

    squared_norms = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = torch.linalg.vector_norm(parameter.grad.detach())
        squared_norms.append(grad_norm.square())
        if grad_norm > max_norm:
            parameter.grad.mul_(max_norm / (grad_norm + 1e-12))
    if not squared_norms:
        return torch.tensor(0.0)
    return torch.sqrt(torch.stack(squared_norms).sum())


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(
        self,
        name,
        model_fn,
        obs_shape_n,
        act_space_n,
        agent_index,
        args,
        local_q_func=False,
        device=None,
        action_spec_n=None,
    ):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.local_q_func = local_q_func
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Official mode derives one categorical branch for physical movement
        # and (when present) one for communication. Gaussian is an explicit
        # PettingZoo-only ablation and is not the original algorithm.
        policy_mode = getattr(args, "policy_mode", "official")
        if action_spec_n is None:
            action_spec_n = [None] * len(act_space_n)
        act_pdtype_n = [
            make_pdtype(
                ac,
                policy_mode=policy_mode,
                branch_sizes=(spec.branch_sizes if spec is not None else None),
            )
            for ac, spec in zip(act_space_n, action_spec_n)
        ]
        obs_dim_n = [int(np.prod(s)) for s in obs_shape_n]
        act_dim_n = [int(pdtype.sample_shape()[0]) for pdtype in act_pdtype_n]
        p_param_size = int(act_pdtype_n[agent_index].param_shape()[0])

        # Actor outputs logits in official mode, or mean/logstd in Gaussian mode.
        self.p_net = MLP(obs_dim_n[agent_index], p_param_size, num_units=args.num_units).to(self.device)
        self.target_p_net = MLP(obs_dim_n[agent_index], p_param_size, num_units=args.num_units).to(self.device)
        if getattr(args, "target_init", "copy") == "copy":
            self.target_p_net.load_state_dict(self.p_net.state_dict())

        # Q 网络：concat(obs_n, act_n) -> Q，或 local 时 concat(obs_i, act_i)
        if local_q_func:
            q_input_dim = obs_dim_n[agent_index] + act_dim_n[agent_index]
        else:
            q_input_dim = sum(obs_dim_n) + sum(act_dim_n)
        self.q_net = MLP(q_input_dim, 1, num_units=args.num_units).to(self.device)
        self.target_q_net = MLP(q_input_dim, 1, num_units=args.num_units).to(self.device)
        if getattr(args, "target_init", "copy") == "copy":
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.act_pdtype_n = act_pdtype_n
        self.action_spec_n = action_spec_n
        self.policy_mode = policy_mode
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.p_optimizer = torch.optim.Adam(self.p_net.parameters(), lr=args.lr)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        # 环境动作空间边界（Box）：用于 tanh 压到 [low, high]，避免 "action outside action space" 告警
        ac_space = act_space_n[agent_index]
        if isinstance(act_pdtype_n[agent_index], DiagGaussianPdType):
            self._action_low = torch.as_tensor(ac_space.low, dtype=torch.float32, device=self.device)
            self._action_high = torch.as_tensor(ac_space.high, dtype=torch.float32, device=self.device)
        else:
            self._action_low = self._action_high = None

    def _squash(self, raw):
        """将无界动作压到 [action_low, action_high]，用 tanh。raw 可为 tensor 或 numpy。"""
        if self._action_low is None:
            return raw
        if isinstance(raw, torch.Tensor):
            x = (torch.tanh(raw) + 1.0) * 0.5 * (self._action_high - self._action_low) + self._action_low
            return x
        raw_t = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        out = (torch.tanh(raw_t) + 1.0) * 0.5 * (self._action_high - self._action_low) + self._action_low
        return out.cpu().numpy()

    def _to_tensor(self, x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=dtype, device=self.device)
        return x.to(self.device)

    def _concat_obs_act(self, obs_n, act_n, use_next_agent_act=None):
        """将 obs_n 与 act_n 按 agent 拼接为 (batch, dim)。local_q 时只用当前 agent。"""
        tensors = []
        if self.local_q_func:
            tensors.append(self._to_tensor(obs_n[self.agent_index]))
            tensors.append(self._to_tensor(act_n[self.agent_index]))
        else:
            for i in range(self.n):
                tensors.append(self._to_tensor(obs_n[i]))
            for i in range(self.n):
                tensors.append(self._to_tensor(act_n[i]))
        return torch.cat(tensors, dim=-1)

    def action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = self._to_tensor(obs).unsqueeze(0)
            p = self.p_net(obs_t)
            act_pd = self.act_pdtype_n[self.agent_index].pdfromflat(p)
            act = act_pd.mode() if deterministic else act_pd.sample()
            act = self._squash(act)
        return act.cpu().numpy().flatten()

    def _target_act(self, obs_batch):
        """用于 target Q 计算：target_p_net(obs) -> sample，并压到动作空间内。"""
        with torch.no_grad():
            p = self.target_p_net(obs_batch)
            act_pd = self.act_pdtype_n[self.agent_index].pdfromflat(p)
            act = act_pd.sample()
            act = self._squash(act)
        return act

    def experience(self, obs, act, rew, new_obs, done, terminal):
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def checkpoint_state(self):
        return {
            "p_net": self.p_net.state_dict(),
            "target_p_net": self.target_p_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "p_optimizer": self.p_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
        }

    def load_checkpoint_state(self, state, load_optimizers=False):
        required = (
            "p_net",
            "target_p_net",
            "q_net",
            "target_q_net",
            "p_optimizer",
            "q_optimizer",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError("trainer checkpoint is missing {}".format(missing))
        self.p_net.load_state_dict(state["p_net"], strict=True)
        self.target_p_net.load_state_dict(state["target_p_net"], strict=True)
        self.q_net.load_state_dict(state["q_net"], strict=True)
        self.target_q_net.load_state_dict(state["target_q_net"], strict=True)
        if load_optimizers:
            self.p_optimizer.load_state_dict(state["p_optimizer"])
            self.q_optimizer.load_state_dict(state["q_optimizer"])

    def preupdate(self):
        """每轮更新前由 train 脚本统一调用：清空缓存的 batch 索引，使本轮各 agent 的 update() 都会重新采样新 batch。"""
        self.replay_sample_index = None

    def _soft_update(self, source, target):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(POLYAK * tp.data + (1.0 - POLYAK) * sp.data)

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return None
        if t % 100 != 0:
            return None

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        index = self.replay_sample_index
        obs_n = []
        obs_next_n = []
        act_n = []
        for i in range(self.n):
            o, a, r, o_next, d = agents[i].replay_buffer.sample_index(index)
            obs_n.append(o)
            obs_next_n.append(o_next)
            act_n.append(a)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # 转为 tensor（batch 在第一维）
        obs_n_t = [self._to_tensor(x) for x in obs_n]
        obs_next_n_t = [self._to_tensor(x) for x in obs_next_n]
        act_n_t = [self._to_tensor(x) for x in act_n]
        rew_t = self._to_tensor(rew)
        done_t = self._to_tensor(done)

        # 所有 agent 的 target 动作（用于 target Q）
        target_act_next_n = []
        for i in range(self.n):
            target_act_next_n.append(agents[i]._target_act(obs_next_n_t[i]))

        # Target Q
        q_next_input = self._concat_obs_act(
            [obs_next_n_t[j] for j in range(self.n)],
            target_act_next_n,
        )
        target_q_next = self.target_q_net(q_next_input).squeeze(-1)
        target_q = bellman_target(
            rew_t, done_t, target_q_next.detach(), self.args.gamma
        )

        # 训练 Q
        q_input = self._concat_obs_act(obs_n_t, act_n_t)
        q = self.q_net(q_input).squeeze(-1)
        q_loss = torch.mean((q - target_q) ** 2)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = clip_grad_norm_per_parameter(
            self.q_net.parameters(), GRAD_NORM_CLIP
        )
        self.q_optimizer.step()

        # 训练 P：用当前 agent 的策略采样动作，其余用 batch 里的动作
        p_input = obs_n_t[self.agent_index]
        p = self.p_net(p_input)
        act_pd = self.act_pdtype_n[self.agent_index].pdfromflat(p)
        act_sample = act_pd.sample()
        act_sample = self._squash(act_sample)
        act_input_n_t = list(act_n_t)
        act_input_n_t[self.agent_index] = act_sample
        q_input_p = self._concat_obs_act(obs_n_t, act_input_n_t)
        self.q_optimizer.zero_grad(set_to_none=True)
        for parameter in self.q_net.parameters():
            parameter.requires_grad_(False)
        try:
            q_p = self.q_net(q_input_p).squeeze(-1)
            p_loss, pg_loss, p_reg = actor_loss(q_p, act_pd.flatparam())
            self.p_optimizer.zero_grad()
            p_loss.backward()
            p_grad_norm = clip_grad_norm_per_parameter(
                self.p_net.parameters(), GRAD_NORM_CLIP
            )
            self.p_optimizer.step()
        finally:
            for parameter in self.q_net.parameters():
                parameter.requires_grad_(True)

        # 动作分布熵与探索程度（用于 TensorBoard）
        action_entropy = torch.mean(act_pd.entropy()).item() if hasattr(act_pd, "entropy") else 0.0
        action_std_mean = (
            torch.mean(act_pd.std).item() if hasattr(act_pd, "std") else 0.0
        )

        # Soft update target
        self._soft_update(self.p_net, self.target_p_net)
        self._soft_update(self.q_net, self.target_q_net)

        return {
            "q_loss": q_loss.item(),                    # Critic 的 TD 均方误差
            "p_loss": p_loss.item(),                   # Actor 总损失 = -Q + 1e-3 * p_reg
            "pg_loss": pg_loss.item(),                 # Actor 策略梯度损失（-mean(Q)）
            "p_reg": p_reg.item(),                     # 策略输出参数的 L2 正则
            "mean_q": float(torch.mean(q).item()),      # 当前 Q(s,a) 的 batch 均值
            "mean_target_q": float(torch.mean(target_q).item()),       # TD 目标 r + gamma*(1-done)*Q' 的均值
            "mean_target_q_next": float(torch.mean(target_q_next).item()),  # 下一状态 target Q 的均值
            "std_target_q": float(torch.std(target_q).item()),         # target_q 的 batch 标准差
            "mean_batch_rew": float(np.mean(rew)),      # 本 batch 奖励均值
            "action_entropy": action_entropy,           # 动作分布熵（探索程度）
            "action_std_mean": action_std_mean,         # 动作分布 std 的均值（探索噪声大小）
            "q_grad_norm": float(q_grad_norm),         # Critic 梯度裁剪前的范数
            "p_grad_norm": float(p_grad_norm),         # Actor 梯度裁剪前的范数
        }
