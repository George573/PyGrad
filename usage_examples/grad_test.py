import numpy as np

from pygrad import Tensor
from pygrad.optimizers.backprop import backward

a = Tensor(np.array([[2.0]]), requires_grad=True)
b = Tensor(np.array([[3.0]]), requires_grad=True)

x = a * b  # x = ab
y = x * a  # y = a²b
z = y + x  # z = a²b + ab

grad_table = backward(z)

print("z:")
print(z.data)

print("grad a:")
print(grad_table[a])

print("grad b:")
print(grad_table[b])

print("grad x:")
print(grad_table[x])
