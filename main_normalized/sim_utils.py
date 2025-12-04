"""Shared helpers for experiment configuration."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - 在部分运行中可能缺少 torch
    torch = None


def resolve_movielens_path(default_path: str | None = None) -> str:
    """Return dataset path, allowing override via ML_DATA_PATH."""
    override = os.environ.get("ML_DATA_PATH")
    if override:
        return override
    if default_path:
        return default_path
    return os.path.join(os.path.dirname(__file__), "ml-100k", "u.data")


def apply_global_seed(seed: Optional[int]) -> None:
    """Seed python, numpy, and torch (if present) for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
