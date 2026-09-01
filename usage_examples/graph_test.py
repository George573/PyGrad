import pygrad
from pygrad.tensor import Tensor

a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)
c = Tensor([7, 8, 9], requires_grad=True)

x1 = a + b
x2 = a + c

x3 = x1 + x2

x4 = x3 + a
x5 = x1 + x3

x6 = x4 + x5

x7 = x6 + x2

out = x7 + x3

pygrad.utils.draw.print_graph(out)

grad_table = pygrad.optimizers.backprop.backward(out)
print(grad_table)
