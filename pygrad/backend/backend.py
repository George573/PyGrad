"""Backend implementation for PyGrad."""

from typing import Literal

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None  # CuPy is optional; handle the case where it's not installed


def is_array(obj):
    return isinstance(obj, np.ndarray) or (
        cp is not None and isinstance(obj, cp.ndarray)
    )


def as_array(obj, device: Literal["cpu", "cuda"] = "cpu"):
    if device == "cpu":
        return np.asarray(obj)

    if device == "cuda":
        if cp is None:
            raise ValueError("CUDA backend requires CuPy. Install pygrad[gpu].")
        return cp.asarray(obj)

    raise ValueError(f"Unknown device: {device}")


def get_array_module(x):
    if isinstance(x, np.ndarray):
        return np

    if cp is not None and isinstance(x, cp.ndarray):
        return cp

    raise TypeError(f"Unknown array type: {type(x)}")
