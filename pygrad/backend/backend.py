"""Backend implementation for PyGrad."""

from typing import Literal

import numpy as np
import cupy as cp

ARRAY_TYPES = (np.ndarray, cp.ndarray)

def is_array(obj):
    return isinstance(obj, ARRAY_TYPES)

def as_array(obj, device: Literal['cpu', 'cuda'] = 'cpu'):
    if device == "cpu":
        return np.asarray(obj)

    if device == "cuda":
        return cp.asarray(obj)

    raise ValueError(f"Unknown device: {device}")