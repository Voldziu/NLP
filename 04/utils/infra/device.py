"""Wybór urządzenia i ziarno losowości."""

import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Zwraca `cuda`, potem `mps` (Mac), na końcu `cpu`.

    `prefer` wymusza konkretne urządzenie, np. `get_device("cpu")` przy debugowaniu.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Ustawia ziarno w `random`, `numpy` i `torch`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
