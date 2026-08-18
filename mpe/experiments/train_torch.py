"""MADDPG PyTorch training entry point.

The default ``legacy + official`` combination follows the archived OpenAI MPE
and TensorFlow MADDPG action semantics. ``pettingzoo`` and ``gaussian`` are
explicit comparison modes and do not share checkpoints with the default mode.

渲染可视化（弹窗看 MPE）：
  python experiments/train_torch.py --display --load-dir <模型根目录> --scenario simple
  实际从 load-dir/maddpg/<backend>/<policy-mode>/<scenario>/ 加载。可加 --eval-episodes N。

评测与胜负指标：
  - 协作场景（如 simple_spread）：总回报 episode_reward 越高越好。
  - 对抗场景（如 simple_tag）：good agents 回报越高越好（少被抓住），adversaries 回报越高越好（多抓住）。
  - TensorBoard 中 reward/episode_reward、reward/agent*_episode_reward 即上述指标。
"""
import argparse
from contextlib import contextmanager
import json
import numpy as np
import time
import pickle
import os
import random
from datetime import datetime

import torch
from maddpg.common.env_adapters_torch import (
    SCENARIO_TO_PETTINGZOO,
    infer_action_specs,
    make_env as make_backend_env,
)
from maddpg.common.scenario_metrics import get_scenario_metric_plugin
from maddpg.common.tensorboard_logger_torch import TensorBoardIntervalLogger
from maddpg.common.tf_util_torch import get_device, load_state, save_state
from maddpg.trainer.maddpg_torch import MADDPGAgentTrainer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="MPE scenario name; simple and simple_spread are distinct")
    parser.add_argument(
        "--env-backend",
        choices=("legacy", "pettingzoo"),
        default="legacy",
        help="legacy reproduces the archived OpenAI MPE; pettingzoo uses the maintained API",
    )
    parser.add_argument(
        "--policy-mode",
        choices=("official", "gaussian"),
        default="official",
        help="official uses the TF implementation's Gumbel-Softmax actions",
    )
    parser.add_argument(
        "--target-init",
        choices=("copy", "independent"),
        default="copy",
        help="copy is the stable DDPG default; independent reproduces the TF1 initialization quirk",
    )
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=200_000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of replay transitions per optimization batch")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument(
        "--tb-log-interval",
        type=int,
        default=10_000,
        help="TensorBoard aggregation window in environment steps; 0 disables TensorBoard",
    )
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="chkpt/",
                        help="checkpoint root; maddpg/backend/policy/scenario subdirectories are added")
    parser.add_argument("--save-rate", type=int, default=10000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="checkpoint root to load; defaults to save-dir")
    # Evaluation / 渲染与评测
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False,
                        help="渲染可视化：弹窗显示 MPE 环境，需配合 --load-dir 加载已训练策略")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="仅在与 --display 同时使用时生效：跑满该数量 episode 后退出并打印评测回报（0=不限制）")
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=10,
        help="independent deterministic episodes evaluated after each checkpoint; 0 disables",
    )
    parser.add_argument(
        "--checkpoint-eval-seed",
        type=int,
        default=10000,
        help="base seed reused for comparable checkpoint evaluations",
    )
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for Python, NumPy, Torch, CUDA, and the first environment reset")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
    return parser.parse_args()


# PettingZoo scenario aliases. ``simple`` must never alias ``simple_spread``.
SCENARIO_TO_ENV = SCENARIO_TO_PETTINGZOO


def make_env(scenario_name, arglist, benchmark=False):
    """Create either the archived OpenAI MPE or the PettingZoo backend."""
    return make_backend_env(
        scenario_name=scenario_name,
        env_backend=arglist.env_backend,
        max_cycles=arglist.max_episode_len,
        benchmark=benchmark,
        display=arglist.display,
        policy_mode=arglist.policy_mode,
    )


def _get_agent_order(env):
    """Parallel env 首次 reset 后返回的 agents 顺序。"""
    obs_dict, _ = env.reset()
    return list(obs_dict.keys())


def get_trainers(
    env,
    agent_list,
    obs_shape_n,
    act_space_n,
    action_spec_n,
    num_adversaries,
    arglist,
    device,
):
    trainers = []
    # MPE 中通常有两类智能体：对抗方(adversaries)与协作方(good agents)，可配置不同策略。
    # 第一类用 --adv-policy（如 maddpg/ddpg），第二类用 --good-policy，故分两段创建。
    for i in range(num_adversaries): # 前 num_adversaries 个智能体（对抗方）
        trainers.append(
            MADDPGAgentTrainer(
                "agent_%d" % i,
                None,
                obs_shape_n,
                act_space_n,
                i,
                arglist,
                local_q_func=(arglist.adv_policy == "ddpg"),
                device=device,
                action_spec_n=action_spec_n,
            )
        )
    for i in range(num_adversaries, len(agent_list)): # 其余智能体（协作方 / good agents）
        trainers.append(
            MADDPGAgentTrainer(
                "agent_%d" % i,
                None,
                obs_shape_n,
                act_space_n,
                i,
                arglist,
                local_q_func=(arglist.good_policy == "ddpg"),
                device=device,
                action_spec_n=action_spec_n,
            )
        )
    return trainers


ALGORITHM_NAME = "maddpg"
CHECKPOINT_VERSION = 4
SUPPORTED_CHECKPOINT_VERSIONS = (2, 3, CHECKPOINT_VERSION)
_CHECKPOINT_V2_METADATA_KEYS = (
    "env_backend",
    "scenario",
    "policy_mode",
    "target_init",
    "action_specs",
)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _state_dir(root, arglist):
    return os.path.join(
        os.fspath(root).rstrip(os.sep),
        ALGORITHM_NAME,
        arglist.env_backend,
        arglist.policy_mode,
        arglist.scenario,
    )


def _legacy_state_dir(root, arglist):
    """Pre-algorithm-directory layout, accepted for restore compatibility."""

    return os.path.join(
        os.fspath(root).rstrip(os.sep),
        arglist.env_backend,
        arglist.policy_mode,
        arglist.scenario,
    )


def _checkpoint_metadata(arglist, action_spec_n):
    return {
        "algorithm": ALGORITHM_NAME,
        "env_backend": arglist.env_backend,
        "scenario": arglist.scenario,
        "policy_mode": arglist.policy_mode,
        "target_init": arglist.target_init,
        # Keep inference-time architecture and environment settings in the
        # checkpoint so play_torch.py only needs a checkpoint path.
        "max_episode_len": int(getattr(arglist, "max_episode_len", 25)),
        "num_adversaries": int(getattr(arglist, "num_adversaries", 0)),
        "good_policy": getattr(arglist, "good_policy", "maddpg"),
        "adv_policy": getattr(arglist, "adv_policy", "maddpg"),
        "num_units": int(getattr(arglist, "num_units", 64)),
        # These do not affect inference, but are required to construct the
        # trainers and make a restored training run fully self-describing.
        "lr": float(getattr(arglist, "lr", 1e-2)),
        "gamma": float(getattr(arglist, "gamma", 0.95)),
        "batch_size": int(getattr(arglist, "batch_size", 1024)),
        "action_specs": [spec.to_dict() for spec in action_spec_n],
    }


def _transition_flags(term_dict, trunc_dict, agent_list, episode_step, max_steps):
    """Separate Bellman termination from episode-control truncation."""

    terminated_n = [bool(term_dict.get(agent, False)) for agent in agent_list]
    episode_done_n = [
        bool(term_dict.get(agent, False) or trunc_dict.get(agent, False))
        for agent in agent_list
    ]
    done = all(episode_done_n) if episode_done_n else False
    terminal = episode_step >= max_steps
    return terminated_n, done, terminal, done or terminal


def _load_trainers_from_checkpoint(
    ckpt, trainers, expected_metadata, load_optimizers
):
    required = (
        "checkpoint_version",
        "metadata",
        "train_step",
        "completed_episodes",
        "trainers",
    )
    missing = (
        list(required)
        if not isinstance(ckpt, dict)
        else [key for key in required if key not in ckpt]
    )
    if missing:
        raise ValueError(
            "unsupported checkpoint: current format requires {}; missing {}".format(
                list(required), missing
            )
        )

    checkpoint_version = ckpt.get("checkpoint_version")
    if checkpoint_version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            "unsupported checkpoint version {}; supported versions are {}".format(
                checkpoint_version, SUPPORTED_CHECKPOINT_VERSIONS
            )
        )
    actual_metadata = ckpt.get("metadata")
    metadata_to_compare = expected_metadata
    if checkpoint_version == 2:
        # Version 2 was emitted by the immediately preceding PyTorch port. It
        # has complete network/optimizer state but predates self-describing
        # training hyperparameters, so validate the fields it did persist.
        metadata_to_compare = {
            key: expected_metadata[key]
            for key in _CHECKPOINT_V2_METADATA_KEYS
        }
    elif checkpoint_version == 3:
        # Version 3 is fully self-describing but predates the explicit
        # algorithm identity field and algorithm-specific save directory.
        metadata_to_compare = {
            key: value
            for key, value in expected_metadata.items()
            if key != "algorithm"
        }
    if actual_metadata != metadata_to_compare:
        raise ValueError(
            "checkpoint metadata mismatch: expected {}, got {}".format(
                metadata_to_compare, actual_metadata
            )
        )

    states = ckpt["trainers"]
    if not isinstance(states, list):
        raise ValueError("unsupported checkpoint: trainers must be a list")
    if len(states) != len(trainers):
        raise ValueError(
            "checkpoint has {} agents but the environment has {}".format(
                len(states), len(trainers)
            )
        )
    for trainer, state in zip(trainers, states):
        trainer.load_checkpoint_state(
            state, load_optimizers=load_optimizers
        )
    return int(ckpt["train_step"]), int(ckpt["completed_episodes"])


def _checkpoint_payload(arglist, action_spec_n, trainers, train_step, completed_episodes):
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "metadata": _checkpoint_metadata(arglist, action_spec_n),
        "train_step": int(train_step),
        "completed_episodes": int(completed_episodes),
        "trainers": [trainer.checkpoint_state() for trainer in trainers],
    }


@contextmanager
def _preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def evaluate_checkpoint(
    ckpt,
    arglist,
    device,
    render=False,
    deterministic=True,
    render_delay=0.0,
    episode_callback=None,
):
    """Evaluate a checkpoint in a separate env without perturbing caller RNG.

    Training checkpoint evaluation uses the defaults. ``play_torch.py`` opts
    into rendering and a per-episode callback while sharing exactly the same
    environment/action/checkpoint semantics.
    """

    evaluation_episodes = getattr(arglist, "checkpoint_eval_episodes", 10)
    evaluation_seed = getattr(arglist, "checkpoint_eval_seed", 10000)
    if evaluation_episodes <= 0:
        return None

    with _preserve_rng_state():
        env = make_backend_env(
            scenario_name=arglist.scenario,
            env_backend=arglist.env_backend,
            max_cycles=arglist.max_episode_len,
            benchmark=False,
            display=render,
            policy_mode=arglist.policy_mode,
        )
        if render and arglist.env_backend == "pettingzoo":
            # PettingZoo throttles inside render(); configure that clock rather
            # than sleeping a second time after every environment step.
            render_fps = (
                max(1, int(round(1.0 / render_delay)))
                if render_delay > 0
                else 0
            )
            env.unwrapped.metadata = dict(env.unwrapped.metadata)
            env.unwrapped.metadata["render_fps"] = render_fps
        try:
            initial_obs, _ = env.reset(seed=evaluation_seed)
            agent_list = list(initial_obs.keys())
            world_agents = list(env.unwrapped.world.agents)
            has_adversaries = any(
                getattr(agent, "adversary", False) for agent in world_agents
            )
            agent_roles = [
                (
                    "adversary"
                    if getattr(agent, "adversary", False)
                    else "good"
                    if has_adversaries
                    else "agent"
                )
                for agent in world_agents
            ]
            obs_shape_n = [env.observation_space(a).shape for a in agent_list]
            act_space_n = [env.action_space(a) for a in agent_list]
            action_spec_n = infer_action_specs(
                env, agent_list, arglist.policy_mode
            )
            num_adversaries = min(
                len(agent_list), arglist.num_adversaries
            )
            trainers = get_trainers(
                env,
                agent_list,
                obs_shape_n,
                act_space_n,
                action_spec_n,
                num_adversaries,
                arglist,
                device,
            )
            _load_trainers_from_checkpoint(
                ckpt,
                trainers,
                _checkpoint_metadata(arglist, action_spec_n),
                load_optimizers=False,
            )

            scenario_metric_plugin = get_scenario_metric_plugin(
                arglist.scenario
            )
            episode_rewards = []
            episode_lengths = []
            agent_episode_rewards = [
                [] for _ in range(len(agent_list))
            ]
            task_metric_values = {}

            for episode_index in range(evaluation_episodes):
                obs_dict, _ = env.reset(
                    seed=evaluation_seed + episode_index
                )
                if render and arglist.env_backend == "legacy":
                    env.render()
                obs_n = [
                    np.asarray(obs_dict[agent]) for agent in agent_list
                ]
                episode_reward = 0.0
                per_agent_reward = [0.0] * len(agent_list)
                episode_step = 0

                while True:
                    action_n = [
                        trainers[i].action(
                            obs_n[i], deterministic=deterministic
                        )
                        for i in range(len(agent_list))
                    ]
                    result = env.step(
                        {
                            agent_list[i]: action_n[i]
                            for i in range(len(agent_list))
                        }
                    )
                    next_obs_dict, reward_dict = result[0], result[1]
                    termination_dict, truncation_dict = result[2], result[3]
                    rewards = [
                        float(reward_dict.get(agent, 0.0))
                        for agent in agent_list
                    ]
                    episode_reward += sum(rewards)
                    for i, reward in enumerate(rewards):
                        per_agent_reward[i] += reward

                    episode_step += 1
                    _, _, _, episode_ended = _transition_flags(
                        termination_dict,
                        truncation_dict,
                        agent_list,
                        episode_step,
                        arglist.max_episode_len,
                    )
                    obs_n = [
                        np.asarray(
                            next_obs_dict.get(agent, obs_n[i]),
                            dtype=np.float32,
                        )
                        for i, agent in enumerate(agent_list)
                    ]
                    if render and arglist.env_backend == "legacy":
                        env.render()
                    if (
                        render
                        and arglist.env_backend == "legacy"
                        and render_delay > 0
                    ):
                        time.sleep(render_delay)
                    if episode_ended:
                        episode_task_metrics = {}
                        if scenario_metric_plugin is not None:
                            for metric_name, metric_value in (
                                scenario_metric_plugin(env).items()
                            ):
                                episode_task_metrics[metric_name] = float(
                                    metric_value
                                )
                                task_metric_values.setdefault(
                                    metric_name, []
                                ).append(float(metric_value))
                        break

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                for i, reward in enumerate(per_agent_reward):
                    agent_episode_rewards[i].append(reward)
                if episode_callback is not None:
                    episode_callback(
                        {
                            "episode_index": episode_index + 1,
                            "seed": evaluation_seed + episode_index,
                            "episode_reward": float(episode_reward),
                            "episode_length": int(episode_step),
                            "agent_episode_rewards": [
                                float(reward) for reward in per_agent_reward
                            ],
                            "agent_names": list(agent_list),
                            "agent_roles": list(agent_roles),
                            "task_metrics": episode_task_metrics,
                        }
                    )

            return {
                "train_step": int(ckpt["train_step"]),
                "completed_episodes": int(ckpt["completed_episodes"]),
                "evaluation_episodes": int(evaluation_episodes),
                "evaluation_seed": int(evaluation_seed),
                "deterministic": bool(deterministic),
                "episode_reward_mean": float(np.mean(episode_rewards)),
                "episode_reward_std": float(np.std(episode_rewards)),
                "episode_length_mean": float(np.mean(episode_lengths)),
                "episode_rewards": [
                    float(reward) for reward in episode_rewards
                ],
                "episode_lengths": [
                    int(length) for length in episode_lengths
                ],
                "agent_episode_rewards": [
                    [float(reward) for reward in rewards]
                    for rewards in agent_episode_rewards
                ],
                "agent_episode_reward_mean": [
                    float(np.mean(rewards))
                    for rewards in agent_episode_rewards
                ],
                "agent_names": list(agent_list),
                "agent_roles": list(agent_roles),
                "task_metrics": {
                    metric_name: float(np.mean(values))
                    for metric_name, values in task_metric_values.items()
                },
            }
        finally:
            env.close()


def _evaluation_tensorboard_metrics(evaluation):
    metrics = {
        "eval/episode_reward_mean": evaluation["episode_reward_mean"],
        "eval/episode_reward_std": evaluation["episode_reward_std"],
        "eval/episode_length_mean": evaluation["episode_length_mean"],
    }
    for index, reward in enumerate(
        evaluation["agent_episode_reward_mean"]
    ):
        metrics["eval/agent{}_episode_reward_mean".format(index)] = reward
    for metric_name, metric_value in evaluation["task_metrics"].items():
        metrics["eval/task_{}".format(metric_name)] = metric_value
    return metrics


def _save_checkpoint_and_evaluate(
    save_dir,
    arglist,
    action_spec_n,
    trainers,
    train_step,
    completed_episodes,
    device,
    tb_logger,
):
    checkpoint_filename = "state_steps_{}.pt".format(int(train_step))
    checkpoint_path = save_state(
        os.path.join(save_dir, checkpoint_filename),
        _checkpoint_payload(
            arglist,
            action_spec_n,
            trainers,
            train_step,
            completed_episodes,
        ),
    )
    print("[Checkpoint] saved: {}".format(os.path.abspath(checkpoint_path)))
    evaluation = None
    if getattr(arglist, "checkpoint_eval_episodes", 10) > 0:
        persisted_checkpoint = load_state(
            checkpoint_path, map_location=device
        )
        evaluation = evaluate_checkpoint(
            persisted_checkpoint, arglist, device
        )
        evaluation_paths = (
            os.path.join(
                save_dir,
                "evaluation_steps_{}.json".format(int(train_step)),
            ),
            os.path.join(save_dir, "evaluation.json"),
        )
        for evaluation_path in evaluation_paths:
            with open(evaluation_path, "w", encoding="utf-8") as stream:
                json.dump(evaluation, stream, ensure_ascii=False, indent=2)
                stream.write("\n")

        print(
            "[Checkpoint Eval] steps: {}, episodes: {}, eval episodes: {}, "
            "mean reward: {:.6f}, std: {:.6f}".format(
                evaluation["train_step"],
                evaluation["completed_episodes"],
                evaluation["evaluation_episodes"],
                evaluation["episode_reward_mean"],
                evaluation["episode_reward_std"],
            )
        )
        tb_logger.write_immediate(
            _evaluation_tensorboard_metrics(evaluation),
            train_step,
            flush=True,
        )
    return checkpoint_path, evaluation


def train(arglist):
    tb_log_interval = getattr(arglist, "tb_log_interval", 10_000)
    if tb_log_interval < 0:
        raise ValueError("--tb-log-interval must be greater than or equal to 0")
    if getattr(arglist, "checkpoint_eval_episodes", 10) < 0:
        raise ValueError(
            "--checkpoint-eval-episodes must be greater than or equal to 0"
        )
    _seed_everything(arglist.seed)
    device = get_device(use_cuda=not arglist.no_cuda)

    # Algorithm/backend/policy/scenario all participate in checkpoint identity.
    save_dir = _state_dir(arglist.save_dir, arglist)
    load_root = arglist.load_dir if arglist.load_dir else arglist.save_dir
    load_dir = _state_dir(load_root, arglist)
    legacy_load_dir = _legacy_state_dir(load_root, arglist)
    using_legacy_load_dir = False
    if not os.path.exists(load_dir) and os.path.exists(legacy_load_dir):
        load_dir = legacy_load_dir
        using_legacy_load_dir = True

    # Both adapters expose the same dict-based parallel API to this loop.
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    scenario_metric_plugin = get_scenario_metric_plugin(arglist.scenario)
    if scenario_metric_plugin is not None:
        print("Scenario metric plugin enabled for {}".format(arglist.scenario))
    obs_dict, _ = env.reset(seed=arglist.seed)
    agent_list = list(obs_dict.keys())
    n_agents = len(agent_list)
    obs_shape_n = []
    act_space_n = []
    for a in agent_list:
        obs_shape_n.append(env.observation_space(a).shape)
        act_space_n.append(env.action_space(a))
    action_spec_n = infer_action_specs(env, agent_list, arglist.policy_mode)

    num_adversaries = min(n_agents, arglist.num_adversaries)
    trainers = get_trainers(
        env,
        agent_list,
        obs_shape_n,
        act_space_n,
        action_spec_n,
        num_adversaries,
        arglist,
        device,
    )
    print("Using good policy {} and adv policy {}".format(arglist.good_policy, arglist.adv_policy))

    train_step = 0
    completed_episodes = 0
    if arglist.display or arglist.restore or arglist.benchmark:
        if using_legacy_load_dir:
            print(
                "[Compatibility] loading pre-algorithm-directory checkpoint "
                "from {}".format(load_dir)
            )
        print("Loading previous state from {}...".format(load_dir))
        ckpt = load_state(load_dir, map_location=device)
        train_step, completed_episodes = _load_trainers_from_checkpoint(
            ckpt,
            trainers,
            _checkpoint_metadata(arglist, action_spec_n),
            load_optimizers=(arglist.restore and not arglist.display and not arglist.benchmark),
        )

    # TensorBoard：仅训练时写日志，评测（display/benchmark）不产生
    tb_log_dir = None
    if (
        not arglist.display
        and not arglist.benchmark
        and tb_log_interval > 0
    ):
        tb_log_dir = os.path.join(
            "runs",
            "{}_{}_{}_{}".format(
                arglist.env_backend,
                arglist.policy_mode,
                arglist.scenario,
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            ),
        )
        print(
            "TensorBoard log_dir: {}".format(os.path.abspath(tb_log_dir))
        )
    elif not arglist.display and not arglist.benchmark:
        print("TensorBoard logging disabled (--tb-log-interval=0)")

    tb_logger = TensorBoardIntervalLogger(
        interval=tb_log_interval,
        initial_step=train_step,
        log_dir=tb_log_dir,
    )

    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(n_agents)]
    final_ep_rewards = []
    final_ep_ag_rewards = []
    agent_info = [[[]]]
    obs_n = [np.asarray(obs_dict[a]) for a in agent_list]
    episode_step = 0
    t_start = time.time()
    last_checkpoint_episode = None

    print("Starting iterations...")
    while True:
        # 固定使用 agent_list 顺序
        action_n = [trainers[i].action(obs_n[i]) for i in range(n_agents)]
        action_dict = {agent_list[i]: action_n[i] for i in range(n_agents)}

        result = env.step(action_dict)
        next_obs_dict, rew_dict, term_dict, trunc_dict = result[0], result[1], result[2], result[3]
        info_dict = result[4] if len(result) > 4 else {}

        next_obs_n = [np.asarray(next_obs_dict.get(a, obs_n[i]), dtype=np.float32) for i, a in enumerate(agent_list)]
        rew_n = [float(rew_dict.get(a, 0.0)) for a in agent_list]
        # Time-limit truncation ends the rollout but must not mask Bellman bootstrap.
        episode_step += 1
        terminated_n, done, terminal, episode_ended = _transition_flags(
            term_dict,
            trunc_dict,
            agent_list,
            episode_step,
            arglist.max_episode_len,
        )

        for i, agent in enumerate(trainers): #把经验写入replay buffer
            if i < len(obs_n) and i < len(next_obs_n):
                agent.experience(
                    obs_n[i],
                    action_n[i],
                    rew_n[i],
                    next_obs_n[i],
                    terminated_n[i],
                    terminal,
                )
        obs_n = next_obs_n

        for i, rew in enumerate(rew_n):
            if i < len(episode_rewards):
                episode_rewards[-1] += rew
            if i < len(agent_rewards):
                agent_rewards[i][-1] += rew

        if episode_ended:
            episode_task_metrics = (
                scenario_metric_plugin(env) if scenario_metric_plugin else {}
            )
            tb_logger.record_episode(
                episode_reward=episode_rewards[-1],
                episode_length=episode_step,
                agent_episode_rewards=(
                    [rewards[-1] for rewards in agent_rewards]
                    if num_adversaries > 0
                    else None
                ),
                task_metrics=episode_task_metrics,
            )
            # display 模式下可选：跑满 --eval-episodes 后退出并打印评测指标
            if arglist.display and arglist.eval_episodes > 0 and len(episode_rewards) >= arglist.eval_episodes:
                n_ep = min(len(episode_rewards), arglist.eval_episodes)
                rews = episode_rewards[-n_ep:] if n_ep > 0 else []
                print("[Eval] {} episodes, mean episode reward: {:.2f}, std: {:.2f}".format(
                    n_ep, np.mean(rews), np.std(rews) if len(rews) > 1 else 0.0))
                if num_adversaries > 0 and agent_rewards:
                    for i in range(n_agents):
                        ar = agent_rewards[i][-n_ep:] if len(agent_rewards[i]) >= n_ep else agent_rewards[i]
                        if ar:
                            print("  agent{} mean reward: {:.2f}".format(i, np.mean(ar)))
                break

        if episode_ended:
            completed_episodes += 1
            obs_dict, _ = env.reset()
            obs_n = [np.asarray(obs_dict[a]) for a in agent_list]
            episode_step = 0
            episode_rewards.append(0.0)
            for a in agent_rewards:
                a.append(0.0)
            agent_info.append([[]])

        train_step += 1

        if arglist.benchmark:
            if train_step > arglist.benchmark_iters and episode_ended:
                os.makedirs(arglist.benchmark_dir, exist_ok=True)
                file_name = os.path.join(arglist.benchmark_dir, (arglist.exp_name or "benchmark") + ".pkl")
                with open(file_name, "wb") as fp:
                    pickle.dump(agent_info[:-1], fp)
                print("Finished benchmarking, saved to", file_name)
                break
            continue

        if arglist.display:
            time.sleep(0.1)
            continue

        update_results = []
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            res = agent.update(trainers, train_step)
            if res is not None:
                update_results.append(res)

        tb_logger.record_training_update(update_results)

        if episode_ended and completed_episodes % arglist.save_rate == 0:
            recent_count = min(arglist.save_rate, len(episode_rewards) - 1)
            recent_episode_rewards = episode_rewards[-recent_count - 1 : -1]
            if num_adversaries == 0:
                print(
                    "steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step,
                        completed_episodes,
                        np.mean(recent_episode_rewards),
                        round(time.time() - t_start, 3),
                    )
                )
            else:
                print(
                    "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step,
                        completed_episodes,
                        np.mean(recent_episode_rewards),
                        [
                            np.mean(rew[-recent_count - 1 : -1])
                            for rew in agent_rewards
                        ],
                        round(time.time() - t_start, 3),
                    )
                )
            _save_checkpoint_and_evaluate(
                save_dir,
                arglist,
                action_spec_n,
                trainers,
                train_step,
                completed_episodes,
                device,
                tb_logger,
            )
            last_checkpoint_episode = completed_episodes
            t_start = time.time()
            mean_rew_save = np.mean(recent_episode_rewards)
            tb_logger.record_latest(
                "reward/episode_reward_mean_save_interval",
                mean_rew_save,
            )
            final_ep_rewards.append(mean_rew_save)
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-recent_count - 1 : -1]))

        tb_logger.maybe_flush(train_step)

        if completed_episodes >= arglist.num_episodes:
            # Always leave an exact final checkpoint, even when num-episodes is
            # not divisible by save-rate.
            if last_checkpoint_episode != completed_episodes:
                _save_checkpoint_and_evaluate(
                    save_dir,
                    arglist,
                    action_spec_n,
                    trainers,
                    train_step,
                    completed_episodes,
                    device,
                    tb_logger,
                )
            os.makedirs(arglist.plots_dir, exist_ok=True)
            exp_name = arglist.exp_name or "exp"
            rew_file_name = os.path.join(arglist.plots_dir, exp_name + "_rewards.pkl")
            with open(rew_file_name, "wb") as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = os.path.join(arglist.plots_dir, exp_name + "_agrewards.pkl")
            with open(agrew_file_name, "wb") as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print("...Finished total of {} episodes.".format(completed_episodes))
            break

    tb_logger.close(train_step)
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    arglist = parse_args()
    train(arglist)
