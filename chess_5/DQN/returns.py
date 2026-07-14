from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool
    discount: float = 1.0


class NStepAccumulator:
    """Build one replay item per raw transition while preserving env boundaries."""

    def __init__(self, num_envs: int, n_step: int, gamma: float) -> None:
        if n_step < 1:
            raise ValueError("n_step must be at least 1")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.buffers: List[Deque[Transition]] = [deque() for _ in range(num_envs)]

    def add(self, env_index: int, transition: Transition) -> List[Transition]:
        if self.n_step == 1:
            # One-step TD 不需要轨迹缓存或聚合循环；bootstrap 系数仍必须是 gamma。
            return [Transition(
                state=transition.state,
                action=transition.action,
                reward=transition.reward,
                next_state=transition.next_state,
                next_mask=transition.next_mask,
                done=transition.done,
                discount=self.gamma,
            )]
        buffer = self.buffers[env_index]
        buffer.append(transition)
        emitted: List[Transition] = []
        if transition.done:
            while buffer:
                emitted.append(self._aggregate(buffer))
                buffer.popleft()
        elif len(buffer) >= self.n_step:
            emitted.append(self._aggregate(buffer))
            buffer.popleft()
        return emitted

    def _aggregate(self, buffer: Deque[Transition]) -> Transition:
        first = buffer[0]
        reward = 0.0
        last: Optional[Transition] = None
        steps = 0
        for item in buffer:
            reward += (self.gamma ** steps) * item.reward
            last = item
            steps += 1
            if item.done or steps >= self.n_step:
                break
        assert last is not None
        return Transition(
            state=first.state,
            action=first.action,
            reward=reward,
            next_state=last.next_state,
            next_mask=last.next_mask,
            done=last.done,
            discount=self.gamma ** steps,
        )


def board_potential(encoded_state: np.ndarray) -> float:
    """Return a bounded black-perspective potential from all length-five lines."""  # 根据所有潜在五连窗口估计当前行棋方的局面势能。
    state = np.asarray(encoded_state)  # 期望形状为 [3,H,W]，三个通道依次表示己方、对手和空位。
    if state.ndim != 3 or state.shape[0] != 3:  # 拒绝带 batch 维或通道数错误的状态，避免静默广播出错。
        raise ValueError(
            f"encoded_state must have shape [3, H, W], got {state.shape}"
        )
    if state.shape[1] != state.shape[2]:  # 当前五子棋环境只支持方形棋盘。
        raise ValueError(f"encoded_state board must be square, got {state.shape[1:]}")
    if not np.any(state):  # 全零数组专门表示终局后不存在可 bootstrap 的状态。
        return 0.0  # 终局势能必须为零，才能满足 potential-based shaping 的边界条件。
    own = state[0].astype(np.int8, copy=False)  # [H,W]：当前行棋方棋子位置的 0/1 平面。
    opponent = state[1].astype(np.int8, copy=False)  # [H,W]：对手棋子位置的 0/1 平面。
    size = own.shape[0]
    weights = np.asarray([0.0, 0.02, 0.08, 0.25, 0.65, 1.0], dtype=np.float32)  # 下标是窗口内同色棋子数。
    total = 0.0
    lines = 0
    for row in range(size):  # 枚举每个可能的五连窗口起点行。
        for col in range(size):  # 枚举每个可能的五连窗口起点列。
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):  # 横、竖、主对角线和副对角线四个方向。
                end_row = row + 4 * dr  # 五个位置中最后一个位置的行坐标。
                end_col = col + 4 * dc  # 五个位置中最后一个位置的列坐标。
                if not (0 <= end_row < size and 0 <= end_col < size):  # 整个五连窗口必须位于棋盘内部。
                    continue  # 越界方向不构成合法窗口，也不计入归一化分母。
                own_count = 0
                opponent_count = 0
                for offset in range(5):  # 统计当前长度为 5 的窗口中双方各自的棋子数量。
                    own_count += int(own[row + offset * dr, col + offset * dc])  # 累加己方棋子数，范围 0～5。
                    opponent_count += int(opponent[row + offset * dr, col + offset * dc])  # 累加对手棋子数，范围 0～5。
                if opponent_count == 0:  # 只有未被对手棋子阻断的窗口才具有己方成五潜力。
                    total += float(weights[own_count])  # 己方棋子越多，该窗口提供的正势能越大。
                if own_count == 0:  # 只有未被己方棋子阻断的窗口才具有对手成五威胁。
                    total -= float(weights[opponent_count])  # 对手棋子越多，该窗口产生的负势能越大。
                lines += 1  # 记录合法五连窗口数量，用于消除不同棋盘尺寸的数量差异。
    return total / max(1, lines)  # 按窗口数归一化，使最终势能大致限制在 [-1,1]。


def shaped_reward(
    reward: float,
    state: np.ndarray,
    next_state: np.ndarray,
    done: bool,
    gamma: float,
    scale: float,
) -> float:
    if scale <= 0:
        return float(reward)
    current_potential = board_potential(state)
    next_potential = 0.0 if done else board_potential(next_state)
    return float(reward + scale * (gamma * next_potential - current_potential))
