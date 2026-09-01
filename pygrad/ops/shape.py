from pygrad.ops.ops import Ops


class Reshape(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.reshape(self.shape)

    def backward(self, grad):
        (a,) = self.inputs
        return (grad.reshape(a.shape),)

    def __call__(self, a, shape):
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
        self.inputs = (a,)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Transpose(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.transpose(self.axes)

    def backward(self, grad):
        if self.axes is None:
            return (grad.transpose(),)
        inverse_axes = tuple(
            sorted(range(len(self.axes)), key=self.axes.__getitem__)
        )
        return (grad.transpose(inverse_axes),)

    def __call__(self, a, axes=None):
        self.inputs = (a,)
        self.axes = axes
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)
