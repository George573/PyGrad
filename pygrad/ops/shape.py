from pygrad.ops.ops import Ops


class Reshape(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.reshape(self.shape)

    def backward(self, grad):
        (a,) = self.inputs
        return (grad.reshape(a.shape),)

    def __call__(self, a, shape):
        super().__call__()
        self.inputs = (a,)
        self.shape = shape
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Flatten(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.flatten()

    def backward(self, grad):
        (a,) = self.inputs
        return (grad.reshape(a.shape),)

    def __call__(self, a):
        super().__call__()
        self.inputs = (a,)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Transpose(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.transpose(self.axes)

    def backward(self, grad):
        if self.axes is None:
            return (grad.transpose(),)
        dims = len(self.axes)
        axes = [ax if ax >= 0 else dims + ax for ax in self.axes]
        inverse_axes = tuple(sorted(range(dims), key=axes.__getitem__))
        return (grad.transpose(inverse_axes),)

    def __call__(self, a, axes=None):
        super().__call__()
        self.inputs = (a,)
        self.axes = axes
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)
