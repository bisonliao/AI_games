from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"


def cpu_state_dict(module: torch.nn.Module) -> dict[str, object]:
    """Return a queue-safe snapshot (NumPy arrays avoid torch shared-FD reducers)."""
    return {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}
