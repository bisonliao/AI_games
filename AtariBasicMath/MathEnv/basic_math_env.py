from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import spaces


ActionMode = Literal["macro", "raw"]


@dataclass(frozen=True)
class BasicMathConfig:
    max_answer: int = 18
    tens_categories: int = 11  # blank followed by digits 0..9
    ones_categories: int = 10
    reset_noops: int = 120
    transition_noops: int = 300
    submit_wait_frames: int = 300
    release_noops: int = 8
    max_primitive_steps: int = 96
    primitive_action_penalty: float = 0.001
    distance_reward_scale: float = 0.1
    goal_tolerance: float = 0.0
    frameskip: int = 1
    repeat_action_probability: float = 0.0
    isolated_question_mode: bool = True
    random_warmup_questions: int = 3
    random_warmup_noops: int = 60

    def __post_init__(self) -> None:
        if self.primitive_action_penalty < 0.0:
            raise ValueError("primitive_action_penalty must be non-negative")
        if self.distance_reward_scale < 0.0:
            raise ValueError("distance_reward_scale must be non-negative")
        if not 0 <= self.random_warmup_questions < 10:
            raise ValueError("random_warmup_questions must be in [0, 9]")
        if self.random_warmup_noops <= 0:
            raise ValueError("random_warmup_noops must be positive")
        if self.random_warmup_questions and not self.isolated_question_mode:
            raise ValueError(
                "random_warmup_questions requires isolated_question_mode=True"
            )


class BasicMathEnv(gym.Env):
    """Randomized one-question BasicMath environment.

    By default, every reset starts a new native game, solves three warmup
    questions after randomized frame delays, and exposes the fourth question as
    the sole Gymnasium episode. Macro mode selects an answer directly. Raw mode
    emits the native six ALE actions and can optionally include a rendered goal.
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 60}

    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5

    CURSOR_RAM = 9
    INPUT_READY_RAM = 28
    OPERAND_TOP_RAM = 46
    OPERAND_BOTTOM_RAM = 47
    ANSWER_FIRST_RAM = 68
    ANSWER_TENS_RAM = 69
    ANSWER_ONES_RAM = 70
    ANSWER_LAST_RAM = 73
    BLANK_DIGIT = 10
    TENS_CURSOR = 1
    ONES_CURSOR = 2
    MIN_CURSOR = 0
    MAX_CURSOR = 5

    def __init__(
        self,
        action_mode: ActionMode = "raw",
        goal_conditioned: bool = False,
        render_mode: Literal["human", "rgb_array"] | None = "rgb_array",
        config: BasicMathConfig | None = None,
    ) -> None:
        super().__init__()
        if action_mode not in {"macro", "raw"}:
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        if action_mode == "macro" and goal_conditioned:
            raise ValueError("goal_conditioned observations are only valid in raw mode")

        self.config = config or BasicMathConfig()
        self.action_mode = action_mode
        self.goal_conditioned = goal_conditioned
        self.render_mode = render_mode

        gym.register_envs(ale_py)
        self._env = self._create_ale_env(render_mode)
        self._goal_env = None
        self._goal_env_initialized = False
        if goal_conditioned:
            self._goal_env = self._create_ale_env("rgb_array")
        meanings = self._env.unwrapped.get_action_meanings()
        expected = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"]
        if meanings != expected:
            raise RuntimeError(f"Unexpected BasicMath action set: {meanings}")

        self.macro_action_space = spaces.Discrete(self.config.max_answer + 1)
        self.raw_action_space = spaces.Discrete(len(expected))
        self.action_space = (
            self.macro_action_space if action_mode == "macro" else self.raw_action_space
        )

        screen_space = spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
        if goal_conditioned:
            self.observation_space = spaces.Dict(
                {
                    "current": screen_space,
                    "goal": screen_space,
                    "macro": spaces.Box(
                        0.0,
                        1.0,
                        shape=(self.macro_vector_dim,),
                        dtype=np.float32,
                    ),
                    "current_answer": spaces.Box(
                        0.0,
                        1.0,
                        shape=(self.current_answer_vector_dim,),
                        dtype=np.float32,
                    ),
                    "cursor": spaces.Box(
                        0.0,
                        1.0,
                        shape=(self.cursor_vector_dim,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = screen_space

        self._initialized = False
        self._underlying_terminal = False
        self._episode_active = False
        self._primitive_steps = 0
        self._question_start_state: Any | None = None
        self._target_macro: int | None = None
        self._target_screen: np.ndarray | None = None
        self._warmup_delays: tuple[int, ...] = ()
        self._episode_problem_operands: tuple[int, int] | None = None
        self._episode_question_index: int | None = None
        self._last_screen = np.zeros(screen_space.shape, dtype=np.uint8)

    @property
    def target_macro_action(self) -> int | None:
        return self._target_macro

    @property
    def macro_vector_dim(self) -> int:
        return self.config.tens_categories + self.config.ones_categories

    @property
    def current_answer_vector_dim(self) -> int:
        return 2 * (self.BLANK_DIGIT + 1)

    @property
    def cursor_vector_dim(self) -> int:
        return 2

    @property
    def current_answer_digits(self) -> tuple[int | None, int | None]:
        tens, ones = self._current_answer_raw()
        return self._decode_digit(tens), self._decode_digit(ones)

    def macro_vector(self, macro_action: int) -> np.ndarray:
        self._validate_macro(macro_action)
        vector = np.zeros(self.macro_vector_dim, dtype=np.float32)
        tens, ones = divmod(macro_action, 10)

        # Tens index 0 means blank; digit d uses index d + 1.
        tens_index = 0 if macro_action < 10 else tens + 1
        vector[tens_index] = 1.0
        vector[self.config.tens_categories + ones] = 1.0
        return vector

    def current_answer_vector(self) -> np.ndarray:
        tens, ones = self._current_answer_raw()
        categories = self.BLANK_DIGIT + 1
        vector = np.zeros(self.current_answer_vector_dim, dtype=np.float32)
        vector[self._digit_vector_index(tens)] = 1.0
        vector[categories + self._digit_vector_index(ones)] = 1.0
        return vector

    def cursor_vector(self) -> np.ndarray:
        cursor = self._cursor_position()
        vector = np.zeros(self.cursor_vector_dim, dtype=np.float32)
        if cursor == self.TENS_CURSOR:
            vector[0] = 1.0
        elif cursor == self.ONES_CURSOR:
            vector[1] = 1.0
        return vector

    def answer_distance(self, macro_action: int | None = None) -> int:
        target = self._target_macro if macro_action is None else int(macro_action)
        if target is None:
            raise RuntimeError("No target macro action is set")
        self._validate_macro(target)
        current_tens, current_ones = self._current_answer_raw()
        target_tens = self.BLANK_DIGIT if target < 10 else target // 10
        target_ones = target % 10
        return self._digit_distance(current_tens, target_tens) + self._digit_distance(
            current_ones,
            target_ones,
        )

    def set_target_macro_action(self, macro_action: int) -> dict[str, np.ndarray]:
        if not self.goal_conditioned:
            raise RuntimeError("This environment was not created with goal_conditioned=True")
        self._validate_macro(macro_action)
        if self._question_start_state is None:
            raise RuntimeError("reset() must be called before setting a macro target")
        self._target_macro = macro_action
        self._target_screen = self.render_macro_target(macro_action)
        return self._observation()

    def render_macro_target(self, macro_action: int) -> np.ndarray:
        """Render the stable, pre-submit image for a macro answer."""
        self._validate_macro(macro_action)
        if self._question_start_state is None:
            raise RuntimeError("reset() must be called before rendering a macro target")

        if self._goal_env is None:
            raise RuntimeError("Macro target rendering requires goal_conditioned=True")
        if not self._goal_env_initialized:
            self._goal_env.reset(seed=0)
            self._goal_env_initialized = True
        self._goal_env.unwrapped.ale.restoreSystemState(self._question_start_state)
        self._type_answer(macro_action, self._goal_env)
        return np.asarray(
            self._goal_env.unwrapped.ale.getScreenRGB(), dtype=np.uint8
        ).copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self._warmup_delays = ()

        # Isolated episodes always start a new native game. The opt-in legacy
        # full-game mode carries the same ROM forward across question resets.
        if (
            self.config.isolated_question_mode
            or seed is not None
            or not self._initialized
            or self._underlying_terminal
            or self._episode_active
        ):
            self._start_rom_game(seed)
        if self.config.isolated_question_mode:
            self._randomize_episode_start()

        self._episode_active = True
        self._primitive_steps = 0
        self._question_start_state = self._env.unwrapped.ale.cloneSystemState()
        self._last_screen = self._screen()
        self._episode_problem_operands = self._problem_operands()
        self._episode_question_index = self._round_number()
        self._target_macro = None
        self._target_screen = None

        options = options or {}
        target = options.get("target_macro_action")
        if self.goal_conditioned:
            if target is None:
                target = int(self.np_random.integers(self.config.max_answer + 1))
            self.set_target_macro_action(int(target))

        info = self._info(success=False)
        return self._observation(), info

    def step(
        self,
        action: int,
        target_macro_action: int | None = None,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if not self._episode_active:
            raise RuntimeError("step() called outside an active episode; call reset()")

        if self.action_mode == "macro":
            return self._step_macro(int(action))
        return self._step_raw(int(action), target_macro_action)

    def _step_macro(self, macro_action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._validate_macro(macro_action)
        self._env.unwrapped.ale.restoreSystemState(self._question_start_state)
        self._type_answer(macro_action)
        if self.config.isolated_question_mode:
            game_reward, next_screen, underlying_terminal = (
                self._submit_and_finish_isolated()
            )
        else:
            game_reward, next_screen, underlying_terminal = self._submit_and_prepare_next()

        self._episode_active = False
        self._underlying_terminal = underlying_terminal
        self._last_screen = next_screen
        success = game_reward > 0.0
        info = self._info(success=success)
        info.update({"macro_action": macro_action, "game_reward": game_reward})
        return self._observation(), float(game_reward), True, False, info

    def _step_raw(
        self,
        action: int,
        target_macro_action: int | None,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if not self.raw_action_space.contains(action):
            raise ValueError(f"Invalid raw action: {action}")
        if target_macro_action is not None:
            if self._primitive_steps != 0:
                raise ValueError("target_macro_action can only be supplied on the first step")
            self.set_target_macro_action(int(target_macro_action))

        self._primitive_steps += 1
        answer_before = self._current_answer_raw()
        distance_before = self.answer_distance() if self.goal_conditioned else None
        action_penalty = (
            self.config.primitive_action_penalty if self.goal_conditioned else 0.0
        )
        if action == self.FIRE:
            visual_goal_reached = self._goal_reached()
            answer_goal_reached = bool(
                distance_before == 0 and self._extra_answer_digits_blank()
            )
            if self.config.isolated_question_mode:
                game_reward, next_screen, underlying_terminal = (
                    self._submit_and_finish_isolated()
                )
            else:
                game_reward, next_screen, underlying_terminal = (
                    self._submit_and_prepare_next()
                )
            success = answer_goal_reached if self.goal_conditioned else game_reward > 0.0
            if self.goal_conditioned:
                if success:
                    reward = 1.0 - action_penalty
                else:
                    remaining_steps = self.config.max_primitive_steps - self._primitive_steps + 1
                    reward = -action_penalty * remaining_steps
            else:
                reward = float(game_reward)
            self._episode_active = False
            self._underlying_terminal = underlying_terminal
            self._last_screen = next_screen
            info = self._info(
                success=success,
                answer_raw=answer_before,
                answer_distance=distance_before,
            )
            info.update(
                {
                    "visual_goal_reached": visual_goal_reached,
                    "answer_goal_reached": answer_goal_reached,
                    "game_reward": game_reward,
                }
            )
            return self._observation(), reward, True, False, info

        observation, game_reward, terminated, truncated, _ = self._env.step(action)
        self._last_screen = np.asarray(observation, dtype=np.uint8).copy()
        self._underlying_terminal = bool(terminated)
        answer_after = self._current_answer_raw()
        distance_after = self.answer_distance() if self.goal_conditioned else None
        distance_delta = (
            int(distance_before - distance_after)
            if distance_before is not None and distance_after is not None
            else 0
        )
        distance_reward = self.config.distance_reward_scale * distance_delta
        reward = distance_reward - action_penalty

        if terminated or truncated:
            self._episode_active = False
            info = self._info(
                success=False,
                answer_raw=answer_after,
                answer_distance=distance_after,
                distance_delta=distance_delta,
                distance_reward=distance_reward,
            )
            info["game_reward"] = float(game_reward)
            return (
                self._observation(),
                reward,
                bool(terminated),
                bool(truncated),
                info,
            )

        if self._primitive_steps >= self.config.max_primitive_steps:
            self._episode_active = False
            if self.config.isolated_question_mode:
                self._underlying_terminal = False
            else:
                _, next_screen, underlying_terminal = self._submit_and_prepare_next()
                self._underlying_terminal = underlying_terminal
                self._last_screen = next_screen
            info = self._info(
                success=False,
                answer_raw=answer_after,
                answer_distance=distance_after,
                distance_delta=distance_delta,
                distance_reward=distance_reward,
            )
            info["timeout"] = True
            return self._observation(), reward, False, True, info

        return (
            self._observation(),
            reward,
            False,
            False,
            self._info(
                success=False,
                answer_raw=answer_after,
                answer_distance=distance_after,
                distance_delta=distance_delta,
                distance_reward=distance_reward,
            ),
        )

    def _start_rom_game(self, seed: int | None) -> None:
        if seed is not None and self._initialized:
            self._env.close()
            self._env = self._create_ale_env(self.render_mode)
        observation, _ = self._env.reset(seed=seed)
        self._last_screen = np.asarray(observation, dtype=np.uint8).copy()
        self._underlying_terminal = False
        for _ in range(self.config.reset_noops):
            observation, _, terminated, truncated, _ = self._env.step(self.NOOP)
            self._last_screen = np.asarray(observation, dtype=np.uint8).copy()
            if terminated or truncated:
                observation, _ = self._env.reset(seed=seed)
                self._last_screen = np.asarray(observation, dtype=np.uint8).copy()
        self._initialized = True
        self._episode_active = False

    def _randomize_episode_start(self) -> None:
        delays: list[int] = []
        for warmup_index in range(self.config.random_warmup_questions):
            delay = int(self.np_random.integers(self.config.random_warmup_noops))
            delays.append(delay)
            for _ in range(delay):
                observation, _, terminated, truncated, _ = self._env.step(self.NOOP)
                if terminated or truncated:
                    raise RuntimeError(
                        "BasicMath terminated during randomized warmup delay"
                    )
                self._last_screen = np.asarray(observation, dtype=np.uint8).copy()

            operands = self._problem_operands()
            self._type_answer(sum(operands))
            _, next_screen, underlying_terminal = self._submit_and_prepare_next()
            if underlying_terminal:
                raise RuntimeError(
                    "BasicMath terminated before randomized warmup reached "
                    f"question {self.config.random_warmup_questions}"
                )
            self._last_screen = next_screen
            self._underlying_terminal = False
            expected_round = warmup_index + 1
            if self._round_number() != expected_round:
                raise RuntimeError(
                    "Randomized warmup advanced to unexpected question index "
                    f"{self._round_number()}, expected {expected_round}"
                )

        self._warmup_delays = tuple(delays)
        if self.config.random_warmup_questions and not self._answer_state_blank():
            raise RuntimeError("Randomized warmup did not reach a blank answer state")
        self._question_start_state = self._env.unwrapped.ale.cloneSystemState()
        self._last_screen = self._screen()
        self._episode_problem_operands = self._problem_operands()
        self._episode_question_index = self._round_number()

    def _create_ale_env(self, render_mode: str | None) -> gym.Env:
        return gym.make(
            "ALE/BasicMath-v5",
            mode=5,
            difficulty=0,
            frameskip=self.config.frameskip,
            repeat_action_probability=self.config.repeat_action_probability,
            render_mode=render_mode,
        )

    def _type_answer(self, answer: int, ale_env: gym.Env | None = None) -> None:
        ale_env = ale_env or self._env
        ones = answer % 10
        self._set_blank_digit(ones, ale_env)
        tens = answer // 10
        if tens:
            self._pulse(self.LEFT, ale_env)
            self._set_blank_digit(tens, ale_env)

    def _set_blank_digit(self, digit: int, ale_env: gym.Env) -> None:
        up_count = digit + 1
        down_count = 10 - digit
        action, count = (self.UP, up_count) if up_count <= down_count else (self.DOWN, down_count)
        for _ in range(count):
            self._pulse(action, ale_env)

    def _pulse(self, action: int, ale_env: gym.Env) -> None:
        ale_env.step(action)
        for _ in range(self.config.release_noops):
            ale_env.step(self.NOOP)

    def _submit_and_wait_for_round_end(self) -> tuple[float, np.ndarray, bool]:
        old_round = self._round_number()
        total_reward = 0.0
        observation, reward, terminated, truncated, _ = self._env.step(self.FIRE)
        total_reward += float(reward)

        for _ in range(self.config.submit_wait_frames):
            if terminated or truncated or self._round_number() != old_round:
                break
            observation, reward, terminated, truncated, _ = self._env.step(self.NOOP)
            total_reward += float(reward)
        else:
            raise RuntimeError(
                "BasicMath did not finish the submitted question within "
                f"{self.config.submit_wait_frames} frames"
            )

        return (
            total_reward,
            np.asarray(observation, dtype=np.uint8).copy(),
            bool(terminated or truncated),
        )

    def _submit_and_finish_isolated(self) -> tuple[float, np.ndarray, bool]:
        return self._submit_and_wait_for_round_end()

    def _submit_and_prepare_next(self) -> tuple[float, np.ndarray, bool]:
        total_reward, observation, underlying_terminal = (
            self._submit_and_wait_for_round_end()
        )

        if not underlying_terminal:
            for _ in range(self.config.transition_noops):
                observation, reward, terminated, truncated, _ = self._env.step(
                    self.NOOP
                )
                total_reward += float(reward)
                if terminated or truncated:
                    underlying_terminal = True
                    break
            if not underlying_terminal and not self._answer_state_blank():
                raise RuntimeError(
                    "BasicMath did not reach a blank answer state after "
                    f"{self.config.transition_noops} settling frames"
                )

        return total_reward, np.asarray(observation, dtype=np.uint8).copy(), underlying_terminal

    def _goal_reached(self) -> bool:
        if not self.goal_conditioned or self._target_screen is None:
            return False
        current = self._last_screen.astype(np.int16)
        target = self._target_screen.astype(np.int16)
        difference = float(np.mean(np.abs(current - target)))
        return difference <= self.config.goal_tolerance

    def _observation(self) -> Any:
        current = self._last_screen.copy()
        if not self.goal_conditioned:
            return current
        if self._target_macro is None or self._target_screen is None:
            raise RuntimeError("Goal-conditioned observation requested without a macro target")
        return {
            "current": current,
            "goal": self._target_screen.copy(),
            "macro": self.macro_vector(self._target_macro),
            "current_answer": self.current_answer_vector(),
            "cursor": self.cursor_vector(),
        }

    def _screen(self) -> np.ndarray:
        return np.asarray(self._env.unwrapped.ale.getScreenRGB(), dtype=np.uint8).copy()

    def _round_number(self) -> int:
        return int(self._env.unwrapped.ale.getRAM()[5])

    def _info(
        self,
        success: bool,
        *,
        answer_raw: tuple[int, int] | None = None,
        answer_distance: int | None = None,
        distance_delta: int = 0,
        distance_reward: float = 0.0,
    ) -> dict[str, Any]:
        raw = self._current_answer_raw() if answer_raw is None else answer_raw
        if answer_distance is None and self._target_macro is not None:
            answer_distance = self._answer_distance_from(raw, self._target_macro)
        return {
            "success": bool(success),
            "question_index": (
                self._episode_question_index
                if self.config.isolated_question_mode
                and self._episode_question_index is not None
                else self._round_number()
            ),
            "primitive_steps": self._primitive_steps,
            "target_macro_action": self._target_macro,
            "action_mode": self.action_mode,
            "current_answer_digits": tuple(self._decode_digit(value) for value in raw),
            "answer_distance": answer_distance,
            "distance_delta": int(distance_delta),
            "distance_reward": float(distance_reward),
            "problem_operands": self._episode_problem_operands,
            "warmup_delays": self._warmup_delays,
        }

    def _ram(self) -> np.ndarray:
        return np.asarray(self._env.unwrapped.ale.getRAM(), dtype=np.uint8)

    def _current_answer_raw(self) -> tuple[int, int]:
        ram = self._ram()
        digits = (int(ram[self.ANSWER_TENS_RAM]), int(ram[self.ANSWER_ONES_RAM]))
        if any(value < 0 or value > self.BLANK_DIGIT for value in digits):
            raise RuntimeError(f"Unexpected BasicMath answer digits in RAM: {digits}")
        return digits

    def _problem_operands(self) -> tuple[int, int]:
        ram = self._ram()
        operands = (
            int(ram[self.OPERAND_TOP_RAM]),
            int(ram[self.OPERAND_BOTTOM_RAM]),
        )
        if not all(1 <= value <= 9 for value in operands):
            raise RuntimeError(f"Unexpected BasicMath addition operands: {operands}")
        return operands

    def _cursor_position(self) -> int:
        cursor = int(self._ram()[self.CURSOR_RAM])
        if cursor < self.MIN_CURSOR or cursor > self.MAX_CURSOR:
            raise RuntimeError(f"Unexpected BasicMath cursor position in RAM: {cursor}")
        return cursor

    def _extra_answer_digits_blank(self) -> bool:
        ram = self._ram()
        return all(
            int(ram[address]) == self.BLANK_DIGIT
            for address in range(self.ANSWER_FIRST_RAM, self.ANSWER_LAST_RAM + 1)
            if address not in {self.ANSWER_TENS_RAM, self.ANSWER_ONES_RAM}
        )

    def _answer_entry_ready(self) -> bool:
        ram = self._ram()
        return (
            int(ram[self.INPUT_READY_RAM]) == 255
            and self._answer_state_blank()
        )

    def _answer_state_blank(self) -> bool:
        return (
            self._current_answer_raw() == (self.BLANK_DIGIT, self.BLANK_DIGIT)
            and self._extra_answer_digits_blank()
            and self._cursor_position() == self.ONES_CURSOR
        )

    @classmethod
    def _decode_digit(cls, value: int) -> int | None:
        return None if value == cls.BLANK_DIGIT else value

    @classmethod
    def _digit_vector_index(cls, value: int) -> int:
        return 0 if value == cls.BLANK_DIGIT else value + 1

    @classmethod
    def _digit_distance(cls, current: int, target: int) -> int:
        categories = cls.BLANK_DIGIT + 1
        return min((target - current) % categories, (current - target) % categories)

    def _answer_distance_from(self, current: tuple[int, int], target: int) -> int:
        target_tens = self.BLANK_DIGIT if target < 10 else target // 10
        target_ones = target % 10
        return self._digit_distance(current[0], target_tens) + self._digit_distance(
            current[1],
            target_ones,
        )

    def _validate_macro(self, macro_action: int) -> None:
        if not self.macro_action_space.contains(macro_action):
            raise ValueError(
                f"Invalid macro action {macro_action}; expected 0..{self.config.max_answer}"
            )

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._screen()
        return self._env.render()

    def close(self) -> None:
        self._env.close()
        if self._goal_env is not None:
            self._goal_env.close()
