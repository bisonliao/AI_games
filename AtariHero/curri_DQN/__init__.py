"""Distributed curriculum Dueling Double DQN for Atari H.E.R.O."""

from .config import TrainConfig
from .model import DuelingDQN

__all__ = ["DuelingDQN", "TrainConfig"]
