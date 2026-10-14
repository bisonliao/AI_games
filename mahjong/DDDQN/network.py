import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int = 306, action_dim: int = 141, hidden_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def raw(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return finite Q values with a legal-action dueling baseline."""
        hidden = self.trunk(obs)
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        # Dueling DQN 的均值聚合：Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)。
        # 直接用 Q=V+A 也能拟合 Q，但 V 加 c、所有 A 减 c 后 Q 不变，
        # 两个分支对公共基准的分工不确定。减均值让有效优势 A-mean(A)
        # 在动作间均值为 0，V 则等于这些动作的 Q 均值（并非 max Q）。
        # 原始 A 仍可整体平移，但不再影响 Q；动作间的相对大小保持不变。
        # 参考原论文公式 (9)：https://proceedings.mlr.press/v48/wangf16.pdf
        if action_mask is None:
            # dim=1 是动作维：对每个样本分别求均值，不跨 batch 平均。
            mean_advantage = advantage.mean(dim=1, keepdim=True)
        else:
            # 本项目只对当前合法动作取均值，避免非法动作输出影响 Q 基准。
            # 终局可能没有合法动作；分母至少为 1 以防除零，此时均值为 0，
            # learner 会另外清零终局的自举项，不在此状态选择动作。
            valid = action_mask.to(dtype=advantage.dtype)
            mean_advantage = (advantage * valid).sum(
                dim=1,
                keepdim=True,
            ) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return value + advantage - mean_advantage

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values = self.raw(obs, action_mask)
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask.bool(), -torch.inf)
        return q_values


def masked_argmax(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    masked = q_values.masked_fill(~action_mask.bool(), -torch.inf)
    return masked.argmax(dim=1)
