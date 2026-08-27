import pygrad.ops.operations as ops
import pygrad.backend as backend


class Tensor:
    """
    A class representing a tensor in PyGrad.
    """

    def __init__(self, data, created=None):
        self.data = backend.as_array(data)
        self.created = created
        
    def __repr__(self):
        return f"Tensor({self.data}) ({hex(id(self))})"
    
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