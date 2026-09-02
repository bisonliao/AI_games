import numpy as np
import torch

from DQN.network import DuelingQNetwork
from DQN.utils import load_state_dict_bytes, serialize_state_dict


def test_network_accepts_uint8_atari_batches() -> None:
    model = DuelingQNetwork((4, 84, 84), 9)
    observations = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    output = model(observations)
    assert output.shape == (3, 9)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_parameter_payload_round_trip() -> None:
    torch.manual_seed(3)
    source = DuelingQNetwork((4, 84, 84), 9)
    payload = serialize_state_dict(source)
    destination = DuelingQNetwork((4, 84, 84), 9)
    destination.load_state_dict(load_state_dict_bytes(payload))

    observations = torch.from_numpy(
        np.zeros((1, 4, 84, 84), dtype=np.uint8)
    )
    torch.testing.assert_close(source(observations), destination(observations))
