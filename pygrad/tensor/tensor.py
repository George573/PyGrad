from typing import Literal

import pygrad.ops.operations as ops
from pygrad import backend


class Tensor:
    """
    A class representing a tensor in PyGrad.
    """

    def __init__(
        self,
        data,
        device: Literal["cpu", "cuda"] = "cpu",
        op=None,
        inputs: list = None,
        outputs: list = None,
    ):
        self.data = backend.as_array(data, device=device)
        self.device = device
        self.op = op
        self.inputs = list() if inputs is None else inputs
        self.outputs = list() if outputs is None else outputs

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor({self.data}) ({hex(id(self))}))"

    def __str__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        if isinstance(other, Tensor):
            return ops.Add()(self, other)
        else:
            raise ValueError("Addition is only supported between Tensors.")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return ops.Sub()(self, other)
        else:
            raise ValueError("Subtraction is only supported between Tensors.")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return ops.Mul()(self, other)
        else:
            raise ValueError("Multiplication is only supported between Tensors.")

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return ops.MatMul()(self, other)
        else:
            raise ValueError("Matrix multiplication is only supported between Tensors.")

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return ops.Div()(self, other)
        else:
            raise ValueError("Division is only supported between Tensors.")

    def __neg__(self):
        return ops.Neg()(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return ops.Pow()(self, other)
        if isinstance(other, int) or isinstance(other, float):
            return ops.Pow()(self, Tensor(other))
        raise ValueError("Exponentiation is only supported between Tensors or scalars.")

    def reshape(self, shape):
        if isinstance(shape, tuple) or isinstance(shape, int):
            return ops.Reshape()(self, shape)
        else:
            raise ValueError("Shape must be a tuple or an integer.")

    def flatten(self):
        return ops.Flatten()(self)

    def transpose(self, axes=None):
        return ops.Transpose()(self, axes)

    def sum(self, axis=None, keepdims=False):
        return ops.Sum()(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return ops.Mean()(self, axis=axis, keepdims=keepdims)
