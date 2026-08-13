import numpy as np

from DQN.trainer import _TensorBoardBuffer, _actor_epsilon_values, _mean_annealed_epsilon


class _Writer:
    def __init__(self):
        self.values = []

    def add_scalar(self, tag, value, step):
        self.values.append((tag, value, step))


def test_tensorboard_buffer_writes_only_on_flush():
    writer = _Writer()
    buffer = _TensorBoardBuffer(writer)
    buffer.mean("train/loss", 1.0)
    buffer.mean("train/loss", 3.0)
    buffer.latest("communication/wait", 5.0)
    assert writer.values == []
    buffer.flush(10_000)
    assert writer.values == [("train/loss", 2.0, 10_000), ("communication/wait", 5.0, 10_000)]


def test_actor_epsilon_values_preserve_range_and_single_actor_exploration():
    assert np.allclose(_actor_epsilon_values(0.05, 0.4, 4), [0.05, 0.1, 0.2, 0.4])
    assert _actor_epsilon_values(0.05, 0.4, 1) == [0.4]


def test_mean_annealed_epsilon_tracks_all_actors():
    starts = [0.05, 0.10, 0.20, 0.40]
    finals = [0.02, 0.04, 0.08, 0.15]
    assert np.isclose(_mean_annealed_epsilon(starts, finals, 0, 1_000), np.mean(starts))
    assert np.isclose(_mean_annealed_epsilon(starts, finals, 500, 1_000), 0.13)
    assert np.isclose(_mean_annealed_epsilon(starts, finals, 2_000, 1_000), np.mean(finals))
