import torch
from torch import nn

from dqn.learner import double_dqn_targets
from dqn.network import DuelingQNetwork


def test_dueling_network_shape_and_centered_advantage():
    network = DuelingQNetwork(5, 3, 16)
    observations = torch.randn(7, 5)
    q_values = network(observations)
    features = network.trunk(observations)
    values = network.value(features)
    assert q_values.shape == (7, 3)
    torch.testing.assert_close(q_values.mean(1, keepdim=True), values)


class FixedNet(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, observations):
        return self.values.expand(len(observations), -1)


def test_double_dqn_selects_online_action_and_evaluates_with_target():
    online = FixedNet([1, 9, 3])
    target = FixedNet([100, 5, 20])
    result = double_dqn_targets(
        online,
        target,
        torch.zeros(2, 5),
        rewards=torch.tensor([1.0, 2.0]),
        discounts=torch.tensor([0.5, 0.0]),
    )
    torch.testing.assert_close(result, torch.tensor([3.5, 2.0]))

