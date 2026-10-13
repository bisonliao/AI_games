from dataclasses import dataclass
import numpy as np

@dataclass
class Transition:
    obs: object
    action: int
    reward: float
    next_obs: object
    terminated: bool
    truncated: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    episode_id: int
    step_index: int
    env_rank: int = 0

def stack_transitions(items):
    if not items:
        return {}
    return {
        'obs': np.asarray([item.obs for item in items]),
        'actions': np.asarray([item.action for item in items], np.int64),
        'rewards': np.asarray([item.reward for item in items], np.float32),
        'next_obs': np.asarray([item.next_obs for item in items]),
        'terminated': np.asarray([item.terminated for item in items], bool),
        'truncated': np.asarray([item.truncated for item in items], bool),
        'action_mask': np.stack([item.action_mask for item in items]),
        'next_action_mask': np.stack(
            [item.next_action_mask for item in items]
        ),
        'episode_id': np.asarray(
            [item.episode_id for item in items], np.int64
        ),
        'step_index': np.asarray(
            [item.step_index for item in items], np.int64
        ),
        'env_rank': np.asarray([item.env_rank for item in items], np.int32),
    }

def validate_transition(t):
    if not 0 <= int(t.action) < 141:
        raise ValueError('action out of range')
    if t.action_mask.shape != (141,) or t.next_action_mask.shape != (141,):
        raise ValueError('mask shape')
    return True
