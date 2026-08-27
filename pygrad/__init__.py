"""PyGrad: lightweight automatic differentiation for Python."""

from . import tensor
from .tensor import Tensor

__version__ = "0.1.0"

__all__ = ["Tensor", "tensor", "__version__"]
