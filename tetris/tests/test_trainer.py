from DQN.trainer import _TensorBoardBuffer


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
