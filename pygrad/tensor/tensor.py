from typing import Literal

from pygrad import backend
from pygrad.ops.arithmetic import Add, Div, MatMul, Mul, Pow, Sub
from pygrad.ops.elementwise import Abs, Exp, Log, Neg, ReLU, Sigmoid, Sqrt, Tanh
from pygrad.ops.reductions import Mean, Sum
from pygrad.ops.shape import Flatten, Reshape, Transpose
from pygrad.backend.backend import is_array


class Tensor:
    """
    A class representing a tensor in PyGrad.
    """

    __array_priority__ = 1000
    
    def __init__(
        self,
        data,
        device: Literal["cpu", "cuda"] = "cpu",
        op=None,
        inputs: list | None = None,
        requires_grad: bool = False,
    ):
        self.data = backend.as_array(data, device=device)
        self.device = device
        self.op = op
        self.inputs = [] if inputs is None else inputs
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def _coerce_operand(self, other):
        if isinstance(other, Tensor):
            if other.device != self.device:
                raise ValueError(
                    f"Cannot operate on tensor from "
                    f"{self.device!r} and {other.device!r}"
                )
            return other

        if isinstance(other, (int, float, complex)):
            return Tensor(other, device=self.device)

        if is_array(other):
            return Tensor(other, device=self.device)

        return NotImplemented

    def __repr__(self):
        return f"Tensor({self.data}) ({hex(id(self))}))"

    def __str__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        other = self._coerce_operand(other)
        if other is NotImplemented:
            return NotImplemented
        return Add()(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce_operand(other)
        if other is NotImplemented:
            return NotImplemented
        return Sub()(self, other)

    def __rsub__(self, other):
        other = self._coerce_operand(other)
        if other is NotImplemented:
            return NotImplemented
        return Sub()(other, self)

    def __mul__(self, other):
        other = self._coerce_operand(other)
        if other is NotImplemented:
            return NotImplemented
        return Mul()(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        other = self._coerce_operand(other)
        return MatMul()(self, other)

    def __truediv__(self, other):
        other = self._coerce_operand(other)
        if other is NotImplemented:
            return NotImplemented
        return Div()(self, other)

    def __rtruediv__(self, other):
        other = self._coerce_operand(other)
        if other is NotImplemented:
            return NotImplemented
        return Div()(other, self)

    def __neg__(self):
        return Neg()(self)

    def __abs__(self):
        return Abs()(self)

    def __pow__(self, other):
        other = self._coerce_operand(other)
        if other is NotImplemented:
            return NotImplemented
        return Pow()(self, other)

    def __rpow__(self, other):
        other = self._coerce_operand(other)
        if other is NotImplemented:
            return NotImplemented
        return Pow()(other, self)

    def reshape(self, shape):
        if isinstance(shape, (tuple, int)):
            return Reshape()(self, shape)
        else:
            raise TypeError("Shape must be a tuple or an integer.")

    def flatten(self):
        return Flatten()(self)

    def transpose(self, axes=None):
        return Transpose()(self, axes)

    def sum(self, axis=None, keepdims=False):
        return Sum()(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return Mean()(self, axis=axis, keepdims=keepdims)

    def exp(self):
        return Exp()(self)

    def log(self):
        return Log()(self)

    def sqrt(self):
        return Sqrt()(self)

    def abs(self):
        return Abs()(self)

    def tanh(self):
        return Tanh()(self)

    def sigmoid(self):
        return Sigmoid()(self)

    def relu(self):
        return ReLU()(self)
