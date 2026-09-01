from pygrad.backend.utils import restore_reduced_dims, unbroadcast
from pygrad.backend.backend import get_array_module
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
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Sub(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data - b.data

    def backward(self, grad):
        a, b = self.inputs
        return (unbroadcast(grad, a.shape), unbroadcast(grad * (-1), b.shape))

    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Mul(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data * b.data

    def backward(self, grad):
        a, b = self.inputs
        grad_a = grad * b.data
        grad_b = grad * a.data
        return (unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape))

    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class MatMul(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data @ b.data

    def backward(self, grad):
        a, b = self.inputs

        a_was_vector = a.ndim == 1
        b_was_vector = b.ndim == 1

        # Promote vectors to matrices.
        a_data = a.data[None, :] if a_was_vector else a.data
        b_data = b.data[:, None] if b_was_vector else b.data

        # Promote output gradient back to matrix form.
        if a_was_vector and b_was_vector:
            grad = grad.reshape(1, 1)
        elif a_was_vector:
            grad = grad[..., None, :]
        elif b_was_vector:
            grad = grad[..., :, None]

        grad_a = grad @ b_data.swapaxes(-1, -2)
        grad_b = a_data.swapaxes(-1, -2) @ grad

        # Undo the temporary vector -> matrix promotion.
        if a_was_vector:
            grad_a = grad_a.squeeze(-2)

        if b_was_vector:
            grad_b = grad_b.squeeze(-1)

        return (
            unbroadcast(grad_a, a.shape),
            unbroadcast(grad_b, b.shape),
        )
        
    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Div(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data / b.data

    def backward(self, grad):
        a, b = self.inputs
        return (
                unbroadcast(grad/b.data, a.shape),
                unbroadcast(-(grad*a.data)/(b.data**2), b.shape)
            )

    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Neg(Ops):
    def forward(self):
        (a,) = self.inputs
        return -a.data

    def backward(self, grad):
        return -grad,

    def __call__(self, a):
        self.inputs = (a,)
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Pow(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data**b.data

    def backward(self, grad):
        a, b = self.inputs
        xp = get_array_module(a.data)
        grad_a = b.data * (a.data ** (b.data - 1)) * grad
        grad_b = (a.data ** b.data) * xp.log(a.data) * grad

        return (
            unbroadcast(grad_a, a.shape),
            unbroadcast(grad_b, b.shape),
        )

    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Reshape(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.reshape(self.shape)

    def backward(self, grad):
        (a,) = self.inputs
        return (grad.reshape(a.shape),)

    def __call__(self, a, b):
        self.inputs = (a,)
        self.shape = b
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Flatten(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.flatten()

    def backward(self, grad):
        (a,) = self.inputs
        return (grad.reshape(a.shape),)

    def __call__(self, a):
        self.inputs = (a,)
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


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
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Sum(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        (a,) = self.inputs
        xp = get_array_module(a.data)

        grad = restore_reduced_dims(
            grad,
            a.data.ndim,
            self.axis,
            self.keepdims
        )

        return (xp.broadcast_to(grad, a.shape),)

    def __call__(self, a, axis=None, keepdims=False):
        self.inputs = (a,)
        self.axis = axis
        self.keepdims = keepdims
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)


class Mean(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.mean(axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        (a,) = self.inputs
        xp = get_array_module(a.data)

        grad = restore_reduced_dims(
            grad,
            a.data.ndim,
            self.axis,
            self.keepdims
        )

        if self.axis is None:
            count = a.data.size
        else:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            count = 1

            for ax in axes:
                count *= a.shape[ax]

        return (
            xp.broadcast_to(grad / count, a.shape),
        )

    def __call__(self, a, axis=None, keepdims=False):
        self.inputs = (a,)
        self.axis = axis
        self.keepdims = keepdims
        result = self.forward()
        return self.create_tensor(result, op=self, input_tensors=self.inputs)
