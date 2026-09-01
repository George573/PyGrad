from pygrad.backend.backend import get_array_module
from pygrad.backend.utils import unbroadcast
from pygrad.ops.ops import Ops


class Add(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data + b.data

    def backward(self, grad):
        a, b = self.inputs
        return (unbroadcast(grad, a.shape), unbroadcast(grad, b.shape))

    def __call__(self, a, b):
        self.inputs = (a, b)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Sub(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data - b.data

    def backward(self, grad):
        a, b = self.inputs
        return (unbroadcast(grad, a.shape), unbroadcast(-grad, b.shape))

    def __call__(self, a, b):
        self.inputs = (a, b)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Mul(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data * b.data

    def backward(self, grad):
        a, b = self.inputs
        return (
            unbroadcast(grad * b.data, a.shape),
            unbroadcast(grad * a.data, b.shape),
        )

    def __call__(self, a, b):
        self.inputs = (a, b)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class MatMul(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data @ b.data

    def backward(self, grad):
        a, b = self.inputs

        a_was_vector = a.ndim == 1
        b_was_vector = b.ndim == 1
        a_data = a.data[None, :] if a_was_vector else a.data
        b_data = b.data[:, None] if b_was_vector else b.data

        if a_was_vector and b_was_vector:
            grad = grad.reshape(1, 1)
        elif a_was_vector:
            grad = grad[..., None, :]
        elif b_was_vector:
            grad = grad[..., :, None]

        grad_a = grad @ b_data.swapaxes(-1, -2)
        grad_b = a_data.swapaxes(-1, -2) @ grad

        if a_was_vector:
            grad_a = grad_a.squeeze(-2)
        if b_was_vector:
            grad_b = grad_b.squeeze(-1)

        return (unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape))

    def __call__(self, a, b):
        self.inputs = (a, b)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Div(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data / b.data

    def backward(self, grad):
        a, b = self.inputs
        return (
            unbroadcast(grad / b.data, a.shape),
            unbroadcast(-(grad * a.data) / (b.data**2), b.shape),
        )

    def __call__(self, a, b):
        self.inputs = (a, b)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Pow(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data**b.data

    def backward(self, grad):
        a, b = self.inputs
        xp = get_array_module(a.data)
        grad_a = b.data * (a.data ** (b.data - 1)) * grad
        grad_b = (a.data**b.data) * xp.log(a.data) * grad
        return (unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape))

    def __call__(self, a, b):
        self.inputs = (a, b)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)
