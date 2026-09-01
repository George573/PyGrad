from pygrad.backend.backend import get_array_module
from pygrad.ops.ops import Ops


class UnaryElementwise(Ops):
    def __call__(self, a):
        self.inputs = (a,)
        return self.create_tensor(self.forward(), op=self, input_tensors=self.inputs)


class Neg(UnaryElementwise):
    def forward(self):
        (a,) = self.inputs
        return -a.data

    def backward(self, grad):
        return (-grad,)


class Exp(UnaryElementwise):
    def forward(self):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return xp.exp(a.data)

    def backward(self, grad):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return (grad * xp.exp(a.data),)


class Log(UnaryElementwise):
    def forward(self):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return xp.log(a.data)

    def backward(self, grad):
        (a,) = self.inputs
        return (grad / a.data,)


class Sqrt(UnaryElementwise):
    def forward(self):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return xp.sqrt(a.data)

    def backward(self, grad):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return (grad / (2 * xp.sqrt(a.data)),)


class Abs(UnaryElementwise):
    def forward(self):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return xp.abs(a.data)

    def backward(self, grad):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return (grad * xp.sign(a.data),)


class Tanh(UnaryElementwise):
    def forward(self):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return xp.tanh(a.data)

    def backward(self, grad):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        output = xp.tanh(a.data)
        return (grad * (1 - output**2),)


class Sigmoid(UnaryElementwise):
    @staticmethod
    def _compute(data):
        xp = get_array_module(data)
        numerator = xp.exp(xp.minimum(data, 0))
        return numerator / (1 + xp.exp(-xp.abs(data)))

    def forward(self):
        (a,) = self.inputs
        return self._compute(a.data)

    def backward(self, grad):
        (a,) = self.inputs
        output = self._compute(a.data)
        return (grad * output * (1 - output),)


class ReLU(UnaryElementwise):
    def forward(self):
        (a,) = self.inputs
        xp = get_array_module(a.data)
        return xp.maximum(a.data, 0)

    def backward(self, grad):
        (a,) = self.inputs
        return (grad * (a.data > 0),)
