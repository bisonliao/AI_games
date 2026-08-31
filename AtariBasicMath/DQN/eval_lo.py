from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DQN.cli import eval_from_args


if __name__ == "__main__":
    eval_from_args("low")
