from dataclasses import dataclass, field
from typing import Literal

@dataclass
class EnvConfig:
    observation_mode: Literal['vector','rgb','both'] = 'vector'
    render_mode: str | None = None
    image_size: tuple[int, int] = (1200, 900)
    seed: int | None = None
    max_steps: int = 1000
    robot_randomness: float = 0.35
    opponent_action_delay: float = 1.5
    win_reward: float = 10.0
    loss_reward: float = -10.0
    draw_reward: float = -1.0
    potential_scale: float = 0.5
    ukeire_weight: float = 0.5
    event_reward: float = 0.0
    include_action_mask_in_obs: bool = True
    metadata: dict = field(default_factory=dict)
