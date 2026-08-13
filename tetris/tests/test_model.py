import numpy as np
import torch

from DQN.model import DuelingDQN, masked_q_values, observations_to_torch


def test_dueling_network_batch_shape():
    obs = {
        "board": np.zeros((2, 20, 10), dtype=np.uint8),
        "active": np.zeros((2, 20, 10), dtype=np.uint8),
        "current_piece": np.zeros((2, 7), dtype=np.int8),
        "next_piece": np.zeros((2, 7), dtype=np.int8),
        "rotation": np.zeros((2, 4), dtype=np.int8),
        "position": np.zeros((2, 2), dtype=np.float32),
    }
    output = DuelingDQN()(observations_to_torch(obs, "cpu"))
    assert output.shape == (2, 40)
    assert torch.isfinite(output).all()


def test_placement_network_and_action_mask():
    obs = {
        "board": np.zeros((2, 20, 10), dtype=np.uint8),
        "active": np.zeros((2, 20, 10), dtype=np.uint8),
        "current_piece": np.eye(7, dtype=np.int8)[:2],
        "next_piece": np.eye(7, dtype=np.int8)[1:3],
        "rotation": np.eye(4, dtype=np.int8)[:2],
        "position": np.zeros((2, 2), dtype=np.float32),
        "action_mask": np.zeros((2, 40), dtype=np.int8),
    }
    obs["action_mask"][0, 3] = 1
    obs["action_mask"][1, 17] = 1
    torch_obs = observations_to_torch(obs, "cpu")
    output = DuelingDQN()(torch_obs)
    masked = masked_q_values(output, torch_obs["action_mask"])
    assert output.shape == (2, 40)
    assert masked.argmax(dim=1).tolist() == [3, 17]


def test_empty_action_mask_uses_action_zero_as_sentinel():
    q_values = torch.tensor([[1.0, 4.0, 3.0]])
    masked = masked_q_values(q_values, torch.zeros_like(q_values, dtype=torch.bool))
    assert masked.argmax(dim=1).item() == 0
    assert masked[0, 0] == q_values[0, 0]
    assert torch.all(masked[0, 1:] == torch.finfo(q_values.dtype).min)
