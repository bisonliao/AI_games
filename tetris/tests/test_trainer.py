from DQN.trainer import _TensorBoardBuffer
from DQN.schedule import epsilon_for_schedule


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


def test_epsilon_switches_all_actor_rates_after_capability_trigger():
    assert epsilon_for_schedule(0.05, False) == 0.05
    assert epsilon_for_schedule(0.10, False) == 0.10
    assert epsilon_for_schedule(0.20, False) == 0.20
    assert epsilon_for_schedule(0.40, False) == 0.40
    assert {
        epsilon_for_schedule(rate, True)
        for rate in (0.05, 0.10, 0.20, 0.40)
    } == {0.05}
