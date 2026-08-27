import pygrad.ops.operations as ops
import pygrad.backend as backend


class Tensor:
    """
    A class representing a tensor in PyGrad.
    """

    def __init__(self, data, created=None):
        self.data = backend.as_array(data)
        self.created = created
        
    def __add__(self, other):
        if isinstance(other, Tensor):
            return ops.Add()(self, other)
        else:
            raise ValueError("Addition is only supported between Tensors.")