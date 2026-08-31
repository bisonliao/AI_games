from __future__ import annotations

import pytest

from MathEnv import BasicMathConfig, BasicMathEnv


def test_macro_action_solves_first_question() -> None:
    env = BasicMathEnv(action_mode="macro")
    try:
        observation, info = env.reset(seed=123)
        assert observation.shape == (210, 160, 3)
        assert info["question_index"] == 3
        assert len(info["warmup_delays"]) == 3
        assert all(0 <= delay < 60 for delay in info["warmup_delays"])

        _, reward, terminated, truncated, info = env.step(
            sum(info["problem_operands"])
        )
        assert reward == 1.0
        assert terminated and not truncated
        assert info["success"] is True
        assert info["question_index"] == 3
    finally:
        env.close()


def test_each_isolated_macro_episode_starts_a_new_randomized_question() -> None:
    env = BasicMathEnv(action_mode="macro")
    try:
        env.reset(seed=7)
        observed_rounds = []
        observed_problems = []
        for _ in range(20):
            _, _, terminated, truncated, info = env.step(0)
            assert terminated and not truncated
            observed_rounds.append(info["question_index"])
            observed_problems.append(info["problem_operands"])
            env.reset()
        assert observed_rounds == [3] * 20
        assert len(set(observed_problems)) >= 10
    finally:
        env.close()


def test_randomized_episode_sequence_is_reproducible_but_not_constant() -> None:
    def collect(seed: int) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
        env = BasicMathEnv(action_mode="macro")
        episodes = []
        try:
            for episode in range(30):
                _, info = env.reset(seed=seed if episode == 0 else None)
                episodes.append((info["problem_operands"], info["warmup_delays"]))
                env.step(0)
        finally:
            env.close()
        return episodes

    first = collect(2345)
    second = collect(2345)
    assert first == second
    assert all(len(delays) == 3 for _, delays in first)
    assert all(0 <= delay < 60 for _, delays in first for delay in delays)
    assert len({problem for problem, _ in first}) >= 15


def test_high_and_low_modes_share_the_same_randomized_problem_distribution() -> None:
    high_env = BasicMathEnv(action_mode="macro")
    low_env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
    try:
        for episode in range(10):
            high_seed = 8765 if episode == 0 else None
            _, high_info = high_env.reset(seed=high_seed)
            _, low_info = low_env.reset(
                seed=high_seed,
                options={"target_macro_action": episode % 19},
            )
            assert high_info["warmup_delays"] == low_info["warmup_delays"]
            assert high_info["problem_operands"] == low_info["problem_operands"]
            assert high_info["question_index"] == low_info["question_index"] == 3
            high_env.step(0)
            low_env.step(low_env.FIRE)
    finally:
        high_env.close()
        low_env.close()


def test_each_randomized_low_episode_is_immediately_editable() -> None:
    env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
    try:
        for episode in range(20):
            observation, info = env.reset(
                seed=4321 if episode == 0 else None,
                options={"target_macro_action": 0},
            )
            assert info["question_index"] == 3
            assert env.current_answer_digits == (None, None)
            assert observation["cursor"].tolist() == [0.0, 1.0]

            _, reward, terminated, truncated, info = env.step(env.UP)
            assert not terminated and not truncated
            assert env.current_answer_digits == (None, 0)
            assert reward == pytest.approx(0.099)
            assert info["answer_distance"] == 0
    finally:
        env.close()


def test_goal_conditioned_raw_action_requires_the_rendered_answer() -> None:
    env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
    try:
        observation, _ = env.reset(seed=123, options={"target_macro_action": 2})
        assert observation["current"].shape == (210, 160, 3)
        assert observation["goal"].shape == (210, 160, 3)
        assert observation["macro"].shape == (21,)
        assert observation["current_answer"].shape == (22,)
        assert observation["cursor"].shape == (2,)
        assert observation["macro"][0] == 1.0  # blank tens digit
        assert observation["macro"][11 + 2] == 1.0
        assert observation["macro"].sum() == 2.0
        assert observation["current_answer"][0] == 1.0
        assert observation["current_answer"][11] == 1.0
        assert observation["current_answer"].sum() == 2.0
        assert observation["cursor"].tolist() == [0.0, 1.0]

        _, reward, terminated, _, info = env.step(env.FIRE)
        assert terminated
        assert reward == -(
            env.config.max_primitive_steps * env.config.primitive_action_penalty
        )
        assert info["visual_goal_reached"] is False
        assert info["answer_goal_reached"] is False

        env.reset(seed=123, options={"target_macro_action": 2})
        for _ in range(3):
            env.step(env.UP)
            for _ in range(env.config.release_noops):
                env.step(env.NOOP)
        _, reward, terminated, _, info = env.step(env.FIRE)
        assert terminated
        assert reward == 1.0 - env.config.primitive_action_penalty
        assert info["visual_goal_reached"] is True
        assert info["answer_goal_reached"] is True
    finally:
        env.close()


def test_macro_vector_factorizes_tens_and_ones() -> None:
    env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
    try:
        env.reset(seed=4, options={"target_macro_action": 12})
        vector = env.macro_vector(12)
        assert vector.shape == (21,)
        assert vector[2] == 1.0  # tens digit 1; index 0 is blank
        assert vector[11 + 2] == 1.0
        assert vector.sum() == 2.0
    finally:
        env.close()


def test_digit_distance_shaping_rewards_progress_and_penalizes_regress() -> None:
    env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
    try:
        observation, info = env.reset(seed=123, options={"target_macro_action": 2})
        assert env.current_answer_digits == (None, None)
        assert info["answer_distance"] == 3

        observation, reward, _, _, info = env.step(env.UP)
        assert reward == pytest.approx(0.099)
        assert info["distance_delta"] == 1
        assert info["distance_reward"] == pytest.approx(0.1)
        assert info["answer_distance"] == 2
        assert env.current_answer_digits == (None, 0)
        assert observation["current_answer"][11 + 1] == 1.0

        _, reward, _, _, info = env.step(env.NOOP)
        assert reward == pytest.approx(-0.001)
        assert info["distance_delta"] == 0

        _, reward, _, _, info = env.step(env.DOWN)
        assert reward == pytest.approx(-0.101)
        assert info["distance_delta"] == -1
        assert info["answer_distance"] == 3

        env.step(env.NOOP)
        observation, reward, _, _, info = env.step(env.LEFT)
        assert reward == pytest.approx(-0.001)
        assert info["distance_delta"] == 0
        assert observation["cursor"].tolist() == [1.0, 0.0]
    finally:
        env.close()


def test_up_and_down_cycle_through_blank_and_all_digits() -> None:
    def collect(action: int) -> list[int | None]:
        env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
        try:
            env.reset(seed=9, options={"target_macro_action": 0})
            values = [env.current_answer_digits[1]]
            for _ in range(11):
                env.step(action)
                values.append(env.current_answer_digits[1])
                env.step(env.NOOP)
            return values
        finally:
            env.close()

    assert collect(BasicMathEnv.UP) == [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None]
    assert collect(BasicMathEnv.DOWN) == [None, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, None]


def test_correct_digits_submit_successfully_with_a_different_cursor_position() -> None:
    env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
    try:
        env.reset(seed=123, options={"target_macro_action": 12})

        # Type the tens digit first and the ones digit second, ending on the
        # ones cursor. The rendered target types in the opposite order and
        # ends on the tens cursor.
        actions = (
            env.LEFT,
            env.NOOP,
            env.UP,
            env.NOOP,
            env.UP,
            env.NOOP,
            env.RIGHT,
            env.NOOP,
            env.UP,
            env.NOOP,
            env.UP,
            env.NOOP,
            env.UP,
            env.NOOP,
        )
        for action in actions:
            env.step(action)

        assert env.current_answer_digits == (1, 2)
        assert env.cursor_vector().tolist() == [0.0, 1.0]
        _, reward, terminated, truncated, info = env.step(env.FIRE)
        assert terminated and not truncated
        assert reward == pytest.approx(0.999)
        assert info["success"] is True
        assert info["answer_goal_reached"] is True
        assert info["visual_goal_reached"] is False
    finally:
        env.close()


def test_cursor_outside_the_two_answer_digits_uses_zero_vector_and_extra_digits_fail() -> None:
    env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
    try:
        env.reset(seed=123, options={"target_macro_action": 0})
        env.step(env.UP)
        env.step(env.NOOP)
        observation, reward, _, _, info = env.step(env.RIGHT)
        assert reward == pytest.approx(-0.001)
        assert info["answer_distance"] == 0
        assert observation["cursor"].tolist() == [0.0, 0.0]

        env.step(env.NOOP)
        env.step(env.UP)  # Fill the first position to the right of the ones digit.
        _, _, terminated, _, info = env.step(env.FIRE)
        assert terminated
        assert info["answer_distance"] == 0
        assert info["answer_goal_reached"] is False
        assert info["success"] is False
    finally:
        env.close()


def test_timeout_keeps_the_last_agent_actions_distance_reward() -> None:
    config = BasicMathConfig(max_primitive_steps=1)
    env = BasicMathEnv(action_mode="raw", goal_conditioned=True, config=config)
    try:
        env.reset(seed=123, options={"target_macro_action": 2})
        _, reward, terminated, truncated, info = env.step(env.UP)
        assert not terminated and truncated
        assert reward == pytest.approx(0.099)
        assert info["timeout"] is True
        assert info["distance_delta"] == 1
        assert info["answer_distance"] == 2
    finally:
        env.close()


def test_each_following_episode_starts_blank_and_editable() -> None:
    env = BasicMathEnv(action_mode="macro")
    try:
        env.reset(seed=41)
        for answer in range(9):
            env.step(answer % 19)
            env.reset()
            assert env.current_answer_digits == (None, None)
            assert env.cursor_vector().tolist() == [0.0, 1.0]
            assert env._answer_entry_ready()
    finally:
        env.close()
