import numpy as np

from PacManEnv.ram_diagnostics import read_ram_fields


def test_reads_only_known_ram_display_fields() -> None:
    ram = np.zeros(128, dtype=np.uint8)
    ram[0] = 1
    ram[119] = 42
    ram[120:124] = [0x10, 0x20, 0x30, 7]

    assert read_ram_fields(ram) == {
        "maze_layout": 1,
        "dots_eaten": 42,
        "score_bcd": [0x10, 0x20, 0x30],
        "life_and_fruit_state": 7,
    }
