from __future__ import annotations

import numpy as np
import torch

from afterBC.common import (
    DEFAULT_BC_CHECKPOINT,
    EXPECTED_BC_SHA256,
    Transition,
    actor_epsilon,
    actor_initial_epsilon,
    file_sha256,
    pack_transitions,
)
from afterBC.network import make_bc_policy, make_dueling_from_bc
from afterBC.replay import (
    PrioritizedReplayBuffer,
    augment_batch,
    transform_action,
)
from afterBC.returns import NStepAccumulator
from afterBC.train import _checkpoint_rank, parse_args, tensorboard_log_dir


def test_bootstrap_checkpoint_has_expected_hash() -> None:
    assert file_sha256(DEFAULT_BC_CHECKPOINT) == EXPECTED_BC_SHA256


def test_tensorboard_run_name_contains_timestamp_and_process_id(tmp_path) -> None:
    result = tensorboard_log_dir(
        tmp_path / "experiment", "white_apex",
        timestamp="20260821_153045", process_id=12345,
    )
    assert result == (
        tmp_path / "experiment" / "tensorboard"
        / "white_apex_20260821_153045_pid12345"
    )


def test_default_update_credit_matches_eight_actor_configuration() -> None:
    args = parse_args(["--run-name", "test"])
    assert args.num_actors == 8
    assert args.actor_torch_threads == 1
    assert args.updates_per_transition == 0.02
    assert 8 * 256 * args.updates_per_transition == 40.96


def test_checkpoint_rank_prioritizes_stochastic_score_over_greedy_audit() -> None:
    greedy_exploit = {
        "deterministic": {"winner": "white"},
        "statistical": {"white_score_rate": 0.4, "white_win_rate": 0.4},
    }
    robust_policy = {
        "deterministic": {"winner": "black"},
        "stochastic": {"white_score_rate": 0.6, "white_win_rate": 0.6},
    }
    assert _checkpoint_rank(robust_policy) > _checkpoint_rank(greedy_exploit)


def test_actor_epsilon_uses_linear_initial_values_and_global_decay() -> None:
    np.testing.assert_allclose(
        [actor_initial_epsilon(index, 4) for index in range(4)],
        [0.4, 0.3, 0.2, 0.1],
    )
    assert actor_initial_epsilon(0, 1) == 0.4
    np.testing.assert_allclose(
        [actor_epsilon(index, 4, 500_000) for index in range(4)],
        [0.225, 0.175, 0.125, 0.075],
    )
    np.testing.assert_allclose(
        [actor_epsilon(index, 4, 1_000_000) for index in range(4)],
        [0.05] * 4,
    )
    assert actor_epsilon(0, 4, 9_000_000) == 0.05


def test_dueling_bc_transfer_preserves_every_greedy_action() -> None:
    policy, _ = make_bc_policy(DEFAULT_BC_CHECKPOINT, device="cpu")
    dueling, _ = make_dueling_from_bc(DEFAULT_BC_CHECKPOINT, device="cpu")
    policy.eval()
    dueling.eval()
    rng = np.random.default_rng(7)
    boards = np.zeros((8, 9, 9), dtype=np.int8)
    players = np.asarray([1, -1] * 4, dtype=np.int8)
    for index in range(1, len(boards)):
        occupied = rng.choice(81, size=index * 2, replace=False)
        boards[index].reshape(-1)[occupied[::2]] = 1
        boards[index].reshape(-1)[occupied[1::2]] = -1
    from afterBC.common import encode_boards
    states = torch.from_numpy(encode_boards(boards, players))
    with torch.no_grad():
        policy_values = policy(states)
        q_values = dueling(states)
    masks = torch.from_numpy(boards.reshape(len(boards), -1) == 0)
    policy_values = policy_values.masked_fill(~masks, -1e9)
    q_values = q_values.masked_fill(~masks, -1e9)
    torch.testing.assert_close(policy_values.argmax(1), q_values.argmax(1))


def test_n_step_terminal_flush_emits_every_white_decision() -> None:
    accumulator = NStepAccumulator(num_envs=1, n_step=3, gamma=0.9)
    board = np.zeros((9, 9), dtype=np.int8)
    mask = np.ones(81, dtype=np.bool_)
    emitted: list[Transition] = []
    for step in range(4):
        emitted.extend(accumulator.add(0, Transition(
            board.copy(), step, 1.0 if step == 3 else 0.0,
            np.zeros_like(board) if step == 3 else board.copy(),
            np.zeros_like(mask) if step == 3 else mask.copy(),
            step == 3,
        )))
    assert len(emitted) == 4
    assert [item.action for item in emitted] == [0, 1, 2, 3]
    np.testing.assert_allclose(
        [item.reward for item in emitted], [0.0, 0.9 ** 2, 0.9, 1.0]
    )
    assert emitted[-1].reward == 1.0
    assert all(item.done for item in emitted[1:])
    assert accumulator.pending() == 0


def test_d4_augmentation_moves_state_action_and_mask_together() -> None:
    state = np.zeros((1, 5, 5), dtype=np.int8)
    state[0, 1, 2] = -1
    next_state = state.copy()
    next_mask = np.zeros((1, 25), dtype=np.bool_)
    next_mask[0, 7] = True
    actions = np.asarray([7])
    transformed = augment_batch(
        state, actions, next_state, next_mask, np.asarray([5])
    )
    out_state, out_actions, out_next, out_mask = transformed
    expected = transform_action(7, 5, 5)
    assert int(out_actions[0]) == expected
    assert out_state.reshape(1, -1)[0, expected] == -1
    assert out_next.reshape(1, -1)[0, expected] == -1
    assert out_mask[0, expected]


def test_prioritized_replay_round_trip_and_priority_update() -> None:
    transitions = []
    for action in range(8):
        board = np.zeros((5, 5), dtype=np.int8)
        board.reshape(-1)[action] = -1
        transitions.append(Transition(
            state=board, action=(action + 1) % 25, reward=float(action == 7),
            next_state=board.copy(), next_mask=board.reshape(-1) == 0,
            done=action == 7, discount=0.99,
        ))
    packet = pack_transitions(
        transitions, actor_id=0, policy_version=2, epsilon=0.2, schedule_step=100
    )
    replay = PrioritizedReplayBuffer(16, 5, seed=3)
    replay.add_packet(packet)
    sample = replay.sample(4, beta=0.4, augment=False)
    assert sample.states.shape == (4, 5, 5)
    assert sample.next_masks.shape == (4, 25)
    assert np.all(sample.weights > 0)
    replay.update_priorities(sample.indices, np.linspace(0.1, 2.0, 4))
    assert replay.max_priority >= 2.0
