"""构建冻结 challenge bank，并对每轮 BC checkpoint 做复合验收。

先理解 challenge bank 在训练流程中的位置：

    Bootstrap 数据生成 -> 冻结 challenge bank -> 训练 round_00 -> 多轮 DAgger

创建 bank 时 Agent 还不存在，因此样本不是由“能力很弱的早期 Agent”产生的。
Bootstrap 棋局来自专家受控采样、局部扰动开局和非战术局面的少量随机动作；所有
动作都通过 GomokuEnv 逐步执行，状态合法，并始终由同一个冻结专家提供 top-1/top-k
监督标签。这里的 OOD 只表示来源不是纯 ``expert_selfplay``，即偏离纯专家路线，
不表示它已经覆盖后续 Agent 可能产生的全部分布外状态。

bank 固定抽取 OOD 状态、win/block/fork 战术状态和非纯专家轨迹的合法行为前缀。
它一旦建立就不能因为 Agent 后来再次访问某个困难状态而删除或替换，否则评测集会
随模型表现变简单，跨轮结果也不再可比。正确做法是保留 challenge 状态，并在每轮
加载 Bootstrap/DAgger 数据时由 train.py 按 canonical key 将它们排除出梯度训练。
因此原始 DAgger NPZ 可能再次包含这些状态，但有效 train split 不会使用它们。

需要注意当前隔离边界：

* 已保证 challenge 中明确保存的 canonical 状态与训练样本互斥，旋转和镜像也视为
  同一个状态；
* 未保证整盘棋或整条 trajectory group 互斥，challenge 前缀附近的其他状态仍可能
  出现在训练数据中；
* ``rollout_audit`` 直接检查本轮 DAgger 数据，它用于诊断当前策略访问分布，但不是
  独立 holdout；若需要严格的动态 OOD 泛化评测，应在每轮生成时按完整轨迹另划一份
  永不训练的 holdout，而不是修改这个固定 bank；
* 当前 build 按 shard 顺序选取最先满足条件的样本，``seed`` 尚未用于随机分层抽样，
  因而标签质量有保障，但样本代表性仍可能受 shard/worker 顺序影响。

评测包含固定状态专家一致率、本轮 rollout audit，以及从固定合法前缀开始、交换
黑白身份的完整对局。由于训练期间每个 epoch 都使用该 bank 选择 ``best.pt``，它在
统计意义上更接近固定验证/模型选择集，而不是训练结束后只使用一次的最终独立测试集。
只有这些复合门槛全部满足时，本轮结果才会标记为 ``passed``。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BC.agent import BCAgent
from BC.heuristic_agent import HeuristicAgent
from BC.oracle import REASON_TO_ID, SOURCE_TO_ID, oracle_identity, validate_oracle_identity
from gomoku_env import GomokuEnv


V3_FIELDS = (
    "boards", "players", "actions", "candidate_actions", "candidate_counts",
    "tactical", "reasons", "plies", "sources", "rounds", "behavior_actions",
    "canonical_keys", "games", "trajectory_groups",
)


def build_bank(data_dir: Path, output: Path, *, seed: int = 0,
               ood_states: int = 20_000, win_states: int = 2_000,
               block_states: int = 2_000, fork_states: int = 1_000,
               prefix_count: int = 500) -> dict[str, Any]:
    """从 Bootstrap 数据中抽取固定 OOD、战术状态和合法对局前缀。

    参数含义：
    ``data_dir`` 是已经 complete 的 v3 Bootstrap 数据集；``output`` 是 challenge
    NPZ 路径；四个 ``*_states`` 分别是 OOD、一步胜、必堵和 fork 的目标数量；
    ``prefix_count`` 是完整对局评测使用的开局前缀数。

    返回值只保存摘要，实际状态和前缀写入 ``output``。注意 ``seed`` 当前保留在
    接口中但没有参与抽样：本函数按 metadata 中的 shard 顺序取最先满足条件的样本。
    """
    # metadata 是数据集的提交清单，提供格式版本、shard 顺序和冻结专家身份。
    metadata = json.loads((data_dir / "metadata.json").read_text())

    # candidate_actions、reasons、sources、canonical_keys 等字段从 v3 才完整存在；
    # 旧数据无法可靠构造当前 challenge，因而直接拒绝。
    if int(metadata.get("format_version", 0)) < 3:
        raise ValueError("challenge banks require v3 datasets")

    # collected 为每个监督状态积累 V3_FIELDS 中全部字段。各列表必须严格等长，
    # 后面会一次性转成 ndarray 写入 NPZ。
    collected = {name: [] for name in V3_FIELDS}

    # prefixes 单独保存用于完整对局的 behavior action 序列，不是监督动作标签。
    prefixes: list[np.ndarray] = []

    # selected_keys 目前只用于阻止 OOD 配额重复计算同一 canonical 状态；
    # 战术样本分支没有用它去重，因此战术配额统计的是样本行数而非唯一状态数。
    selected_keys: set[bytes] = set()

    # 三类战术分别计数：fork 同时包含 own_fork 和 block_fork。
    tactical_counts = {"win": 0, "block": 0, "fork": 0}

    # ood_count 只统计非纯专家来源且 canonical key 未重复的状态。
    ood_count = 0

    # 严格按 metadata 给出的顺序扫描 shard；达到目标后仍会继续遍历文件，
    # 但不再收集已满类别的普通样本。
    for shard_name in metadata["shards"]:
        # np.load 的上下文结束后底层文件会关闭，所以选中的单行字段必须 copy。
        with np.load(data_dir / shard_name) as data:
            # trajectory_groups 在这里主要提供样本数；当前代码没有按 group 划分 bank。
            groups = np.asarray(data["trajectory_groups"])

            # 挑战集 key 会被训练器显式排除，因此这里可从所有已生成轨迹取样，
            # 不必再次沿用普通 train/val split。val 当前全为 True，是为可能的筛选
            # 掩码保留的结构；因此后面的 val[start] 和 flatnonzero(val) 不会过滤样本。
            val = np.ones(len(groups), dtype=bool)

            # generate.py 保证同一 game id 的状态在 shard 内连续排列。
            game_ids = np.asarray(data["games"])

            # 找到每个 game id 发生变化的位置，加上 0 得到每盘棋的起始下标。
            starts = np.r_[0, np.flatnonzero(game_ids[1:] != game_ids[:-1]) + 1]

            # 下一盘的 start 就是上一盘的 end；最后一盘以 shard 样本总数结尾。
            ends = np.r_[starts[1:], len(game_ids)]

            # 前缀必须来自非纯专家轨迹，才能测试策略处理偏离标准路线后的完整对局。
            for start, end in zip(starts, ends):
                # 四个条件依次保证：前缀配额未满、轨迹被允许、至少有 4 手、
                # 该盘的行为来源不是 expert_selfplay。一个 game 的 source 在生成时固定。
                if (len(prefixes) < prefix_count and val[start] and end - start >= 4
                        and int(data["sources"][start]) != SOURCE_TO_ID["expert_selfplay"]):
                    # 设计意图看起来是让前缀长度在 4～16 手之间变化；但当前表达式
                    # len(prefixes) * 13 % 13 恒为 0，所以实际 length 始终为 4
                    # （只要该轨迹至少 4 手）。这里仅说明现状，不在注释修改中改逻辑。
                    length = min(16, max(4, 4 + (len(prefixes) * 13) % 13), end - start)

                    # 必须保存 behavior_actions：它们才是历史上真正推进环境、能够合法
                    # 重放到同一局面的动作；专家 actions 只是监督标签，未必实际落下。
                    prefixes.append(np.asarray(data["behavior_actions"][start:start + length],
                                               dtype=np.int16))

            # 固定状态同时覆盖去重 OOD 与关键战术；一个状态可同时满足两类计数。
            # val 当前全 True，因此按 shard 原始顺序检查每一个状态。
            for index in np.flatnonzero(val):
                # canonical key 已包含当前行动方，并合并旋转/镜像等价棋盘。
                key = bytes(data["canonical_keys"][index])

                # reason 是冻结专家给出的决策原因 ID，用于判断战术类别。
                reason = int(data["reasons"][index])

                # source 描述产生该状态的行为策略，而不是监督标签来自谁；标签始终来自专家。
                source = int(data["sources"][index])

                # win 和 block 要求 top-1 精确正确；own_fork/block_fork 合并为 fork 类；
                # 普通位置和中心开局得到 None，不占战术配额。
                category = ("win" if reason == REASON_TO_ID["win"] else
                            "block" if reason == REASON_TO_ID["block"] else
                            "fork" if reason in (REASON_TO_ID["own_fork"],
                                                 REASON_TO_ID["block_fork"]) else None)

                # 只有属于战术类别且对应目标尚未填满时，才因战术原因收录该状态。
                # 这里没有检查 selected_keys，所以重复 canonical 战术状态也可能被收录。
                want_tactical = bool(
                    category == "win" and tactical_counts["win"] < win_states
                    or category == "block" and tactical_counts["block"] < block_states
                    or category == "fork" and tactical_counts["fork"] < fork_states
                )

                # OOD 在本项目中的操作性定义：来源不是纯 expert_selfplay。
                # 同时要求 canonical key 尚未收录且 OOD 配额未满，保证 OOD 部分去重。
                want_ood = (key not in selected_keys
                            and source != SOURCE_TO_ID["expert_selfplay"]
                            and ood_count < ood_states)

                # 既不能补充战术配额、也不能补充 OOD 配额的状态无需写入 bank。
                if not want_tactical and not want_ood:
                    continue

                # 保存完整 v3 行，而不只保存 board/action；后续评测还需要候选排名、
                # reason、source 和 canonical key。copy 使数组脱离即将关闭的 NPZ。
                for name in V3_FIELDS:
                    collected[name].append(np.asarray(data[name][index]).copy())

                # 无论因 OOD、战术还是两者同时入选，都记录该 canonical key。
                selected_keys.add(key)

                # 同一个状态若同时满足 OOD 和战术，会分别推进两个配额，但只写入一行。
                if want_ood:
                    ood_count += 1
                if want_tactical and category is not None:
                    tactical_counts[category] += 1

    # challenge 必须一次性达到配置规模；缺项时直接失败，不能用缩水测试集放宽门槛。
    # tactical_counts 与 ood_count 可以因同一行同时入选而重叠，所以最终样本总数
    # 通常小于四类目标数量的简单求和。
    if (ood_count < ood_states or len(prefixes) < prefix_count
            or tactical_counts["win"] < win_states
            or tactical_counts["block"] < block_states
            or tactical_counts["fork"] < fork_states):
        # 错误消息同时报告实际值和目标值，便于判断 Bootstrap 缺的是哪类覆盖。
        raise RuntimeError(
            f"bootstrap data cannot fill challenge bank: ood={ood_count}/{ood_states}, "
            f"prefixes={len(prefixes)}/{prefix_count}, tactical={tactical_counts}"
        )

    # 只有全部配额满足后才创建目标目录，失败不会留下看似有效的 challenge NPZ。
    output.parent.mkdir(parents=True, exist_ok=True)

    # NPZ 需要规则矩阵，先用 -1 padding 构造 [prefix_count, 16] 动作数组；
    # -1 不是合法动作，只用于标记每行有效前缀结束后的空槽。
    prefix_array = np.full((prefix_count, 16), -1, dtype=np.int16)

    # 另存真实长度，评测重放时只读取 prefix[:length]，不会执行 -1 padding。
    prefix_lengths = np.zeros(prefix_count, dtype=np.int8)

    # 当前 prefixes 不会超过 prefix_count；切片是额外防御，避免未来收集逻辑变化。
    for index, prefix in enumerate(prefixes[:prefix_count]):
        # 将变长行为序列拷贝到定宽矩阵左侧。
        prefix_array[index, :len(prefix)] = prefix
        # 保存这一行实际可重放的动作数量。
        prefix_lengths[index] = len(prefix)

    # 将逐行积累的 Python list 转成批量 ndarray；字段第一维均为 challenge 样本数。
    arrays = {name: np.asarray(values) for name, values in collected.items()}

    # 状态级监督字段和完整对局前缀共存在同一个 challenge NPZ 中。
    arrays.update({"prefix_actions": prefix_array, "prefix_lengths": prefix_lengths})

    # NPZ 原子提交，旁边的 JSON 保存规模与专家身份，供后续一致性校验。
    # 临时文件使用 .npz 后缀，避免 np.savez_compressed 自动追加后缀导致路径不一致。
    temporary = output.with_suffix(".tmp.npz")

    # 先完整压缩写入临时文件，再用 os.replace 原子替换正式 bank。
    np.savez_compressed(temporary, **arrays); os.replace(temporary, output)

    # 摘要 JSON 不重复保存大数组，只记录来源、规模和 oracle 身份。
    result = {
        "format_version": 1, "source_data": str(data_dir.resolve()),
        "samples": len(arrays["actions"]), "ood_states": ood_count,
        "tactical_counts": tactical_counts, "prefixes": prefix_count,
        "oracle": metadata["oracle"],
    }

    # challenge_v1.json 与 challenge_v1.npz 同名，evaluate_composite 会读取它校验专家版本。
    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # pipeline 将摘要打印到日志；真正评测仍从 output NPZ 读取状态和前缀。
    return result


def _agreement(agent: BCAgent, bank: Path, *, limit: int = 0) -> dict[str, Any]:
    """计算 greedy 策略与冻结专家 top-1/top-4 的状态级一致率。

    bank 既可为固定 challenge NPZ，也可为本轮 DAgger 数据目录；后者用于 rollout audit。
    """
    paths = [bank]
    if bank.is_dir():
        metadata = json.loads((bank / "metadata.json").read_text())
        paths = [bank / name for name in metadata["shards"]]
    fields: dict[str, list[np.ndarray]] = {name: [] for name in
                                           ("boards", "players", "actions",
                                            "candidate_actions", "reasons", "sources")}
    remaining = limit if limit > 0 else None
    for path in paths:
        with np.load(path) as data:
            count = len(data["actions"]) if remaining is None else min(remaining, len(data["actions"]))
            for name in fields:
                fields[name].append(np.asarray(data[name][:count]))
        if remaining is not None:
            remaining -= count
            if remaining <= 0:
                break
    boards = np.concatenate(fields["boards"]); players = np.concatenate(fields["players"])
    targets = np.concatenate(fields["actions"])
    candidates = np.concatenate(fields["candidate_actions"])
    reasons = np.concatenate(fields["reasons"])
    sources = np.concatenate(fields["sources"])
    count = len(targets); masks = boards.reshape(count, -1) == 0
    # 分块推理限制内存；select_actions 会 mask 非法位置并执行 argmax。
    predictions = []
    for start in range(0, count, 2048):
        predictions.append(agent.select_actions(boards[start:start + 2048],
                                                players[start:start + 2048],
                                                masks[start:start + 2048]))
    predictions = np.concatenate(predictions)
    # top-1 衡量是否复制专家首选；top-4 衡量是否至少落在专家认可集合中。
    top1 = predictions == targets
    top4 = ((predictions[:, None] == candidates) & (candidates >= 0)).any(1)
    win_block = np.isin(reasons, [REASON_TO_ID["win"], REASON_TO_ID["block"]])
    forks = np.isin(reasons, [REASON_TO_ID["own_fork"], REASON_TO_ID["block_fork"]])
    ood = sources != SOURCE_TO_ID["expert_selfplay"]
    return {
        "samples": count, "illegal_actions": int(np.count_nonzero(~masks[np.arange(count), predictions])),
        "top1_accuracy": float(top1.mean()) if count else 0.0,
        "top4_accuracy": float(top4.mean()) if count else 0.0,
        "ood_top1_accuracy": float(top1[ood].mean()) if ood.any() else 0.0,
        "ood_top4_accuracy": float(top4[ood].mean()) if ood.any() else 0.0,
        "ood_samples": int(ood.sum()),
        "win_block_accuracy": float(top1[win_block].mean()) if win_block.any() else 0.0,
        "fork_top4_accuracy": float(top4[forks].mean()) if forks.any() else 0.0,
        "win_block_samples": int(win_block.sum()), "fork_samples": int(forks.sum()),
    }


def _match_worker(task: dict[str, Any]) -> dict[str, int]:
    """在一组固定前缀上执行 Agent 对专家的双颜色完整对局。"""
    import torch
    torch.set_num_threads(1)
    agent = BCAgent(task["board_size"], device="cpu")
    agent.load_checkpoint(Path(task["checkpoint"])); agent.net.eval()
    expert = HeuristicAgent(seed=task["seed"], max_candidates=task["max_candidates"])
    totals = {color: {name: 0 for name in ("wins", "losses", "draws", "games")}
              for color in ("black", "white")}
    moves = illegal = 0
    for prefix, length in zip(task["prefixes"], task["lengths"]):
        # 每条前缀复用两次，Agent 分别执黑和执白，消除先后手优势造成的误判。
        for agent_player in (1, -1):
            env = GomokuEnv(board_size=task["board_size"], starting_player="black",
                            illegal_action_mode="raise")
            obs, _ = env.reset(seed=task["seed"]); done = False
            # 先原样重放 behavior action 前缀，再由双方从同一局面继续下完。
            for action in prefix[:int(length)]:
                obs, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    done = True; break
            while not done:
                player = int(obs["current_player"][0])
                if player == agent_player:
                    action = int(agent.select_actions(obs["board"], [player],
                                                      obs["action_mask"])[0])
                else:
                    action = expert.ranked_decision(
                        obs["board"], player, obs["action_mask"], top_k=1
                    ).actions[0]
                if not obs["action_mask"][action]:
                    illegal += 1
                obs, _, terminated, truncated, info = env.step(action)
                moves += 1; done = terminated or truncated
            winner = int(info["winner"])
            color = "black" if agent_player == 1 else "white"
            totals[color]["wins"] += int(winner == agent_player)
            totals[color]["losses"] += int(winner == -agent_player)
            totals[color]["draws"] += int(winner == 0)
            totals[color]["games"] += 1; env.close()
    return {"colors": totals, "moves": moves, "illegal_actions": illegal}


def _matches(checkpoint: Path, bank: Path, board_size: int, workers: int,
             max_candidates: int, seed: int) -> dict[str, Any]:
    """并行完整对局，并计算黑白分色得分率、置信区间和决胜局比例。"""
    with np.load(bank) as data:
        prefixes = data["prefix_actions"]; lengths = data["prefix_lengths"]
    workers = min(workers, len(prefixes))
    chunks = np.array_split(np.arange(len(prefixes)), workers)
    tasks = [{"checkpoint": str(checkpoint), "board_size": board_size,
              "prefixes": prefixes[chunk], "lengths": lengths[chunk],
              "max_candidates": max_candidates, "seed": seed + i * 1_000_003}
             for i, chunk in enumerate(chunks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_match_worker, tasks))
    colors = {}
    for color in ("black", "white"):
        totals = {key: sum(part["colors"][color][key] for part in parts)
                  for key in ("wins", "losses", "draws", "games")}
        # score 将胜/和/负记为 1/0.5/0；置信区间使用样本方差的正态近似。
        score = (totals["wins"] + 0.5 * totals["draws"]) / max(1, totals["games"])
        variance = ((totals["wins"] * (1 - score) ** 2
                     + totals["draws"] * (0.5 - score) ** 2
                     + totals["losses"] * score ** 2) / max(1, totals["games"] - 1))
        margin = 1.959963984540054 * math.sqrt(variance / max(1, totals["games"]))
        colors[color] = {**totals, "score_rate": score,
                         "score_rate_ci95": [max(0.0, score - margin),
                                             min(1.0, score + margin)],
                         "decisive_game_rate":
                             (totals["wins"] + totals["losses"]) / max(1, totals["games"])}
    return {"colors": colors, "moves": sum(part["moves"] for part in parts),
            "illegal_actions": sum(part["illegal_actions"] for part in parts)}


def evaluate_composite(checkpoint: Path, bank: Path, *, board_size: int = 9,
                       audit_data: Path | None = None, workers: int = 16,
                       max_candidates: int = 12, seed: int = 20_000,
                       skip_matches: bool = False) -> dict[str, Any]:
    """执行状态一致率、rollout audit 和完整对局组成的最终质量门。"""
    # 数据、checkpoint 和评测必须声明同一个冻结专家 identity，禁止跨版本比较。
    metadata = json.loads(bank.with_suffix(".json").read_text())
    validate_oracle_identity(metadata.get("oracle"),
                             oracle_identity(max_candidates=max_candidates, top_k=4))
    agent = BCAgent(board_size, device="cpu"); checkpoint_data = agent.load_checkpoint(checkpoint)
    if checkpoint_data.get("oracle") is not None:
        validate_oracle_identity(checkpoint_data["oracle"], metadata["oracle"])
    agreement = _agreement(agent, bank)
    audit = _agreement(agent, audit_data, limit=20_000) if audit_data is not None else None
    matches = (None if skip_matches else
               _matches(checkpoint, bank, board_size, workers, max_candidates, seed))
    # 所有条件是 AND：非法动作必须为零，OOD/战术/rollout 达标，并且 Agent
    # 执黑、执白都要满足得分、置信下界和决胜局比例。全和棋不能伪装成高水平。
    passed = (
        agreement["illegal_actions"] == 0
        and agreement["ood_top1_accuracy"] >= 0.85
        and agreement["ood_top4_accuracy"] >= 0.97
        and agreement["win_block_accuracy"] >= 0.995
        and agreement["fork_top4_accuracy"] >= 0.98
        and (audit is None or audit["top4_accuracy"] >= 0.95)
        and (matches is None or (matches["illegal_actions"] == 0 and all(
            result["score_rate"] >= 0.45
            and result["score_rate_ci95"][0] >= 0.40
            and result["decisive_game_rate"] >= 0.20
            for result in matches["colors"].values()
        )))
    )
    return {"checkpoint": str(checkpoint), "challenge_bank": str(bank),
            "agreement": agreement, "rollout_audit": audit,
            "matches": matches, "passed": bool(passed)}


def parse_args() -> argparse.Namespace:
    """解析 challenge bank 的 build/evaluate 两个子命令。"""
    parser = argparse.ArgumentParser(description="Build/evaluate the fixed BC challenge bank.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--ood-states", type=int, default=20_000)
    build.add_argument("--win-states", type=int, default=2_000)
    build.add_argument("--block-states", type=int, default=2_000)
    build.add_argument("--fork-states", type=int, default=1_000)
    build.add_argument("--prefix-count", type=int, default=500)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--bank", type=Path, required=True)
    evaluate.add_argument("--audit-data", type=Path, default=None)
    evaluate.add_argument("--board-size", type=int, default=9)
    evaluate.add_argument("--workers", type=int, default=16)
    evaluate.add_argument("--max-candidates", type=int, default=12)
    evaluate.add_argument("--seed", type=int, default=20_000)
    evaluate.add_argument("--skip-matches", action="store_true")
    evaluate.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """构建挑战集或评测 checkpoint，并按需将完整结果写入 JSON。"""
    args = parse_args()
    if args.command == "build":
        result = build_bank(args.data_dir, args.output, seed=args.seed,
                            ood_states=args.ood_states, win_states=args.win_states,
                            block_states=args.block_states, fork_states=args.fork_states,
                            prefix_count=args.prefix_count)
    else:
        result = evaluate_composite(
            args.checkpoint, args.bank, board_size=args.board_size,
            audit_data=args.audit_data, workers=args.workers,
            max_candidates=args.max_candidates, seed=args.seed,
            skip_matches=args.skip_matches,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
