from pygrad.backend.backend import get_array_module
from pygrad.backend.utils import restore_reduced_dims
from pygrad.ops.ops import Ops


class Sum(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        grad = restore_reduced_dims(grad, a.ndim, self.axis, self.keepdims)
        return (xp.broadcast_to(grad, a.shape),)

    def __call__(self, a, axis=None, keepdims=False):
        self.inputs = (a,)
        self.axis = axis
        self.keepdims = keepdims
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Mean(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.mean(axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        grad = restore_reduced_dims(grad, a.ndim, self.axis, self.keepdims)

        if self.axis is None:
            count = a.size
        else:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            count = 1
            for axis in axes:
                count *= a.shape[axis]

        return (xp.broadcast_to(grad / count, a.shape),)

    def __call__(self, a, axis=None, keepdims=False):
        self.inputs = (a,)
        self.axis = axis
        self.keepdims = keepdims
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)
