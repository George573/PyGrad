import numpy as np

from pygrad import Tensor
from pygrad.optimizers.backprop import backward


a = Tensor(
    np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ]),
    requires_grad=True,
)

b = Tensor(
    np.array([
        [2.0, 0.0],
        [1.0, 3.0]
    ]),
    requires_grad=True,
)

# Shared intermediate
x = a @ b

# x is used twice
y = x @ b
z = x + y

# Reduce to scalar using matmul only
left = Tensor(np.array([[1.0, 1.0]]))      # (1, 2)
right = Tensor(np.array([[1.0], [1.0]]))   # (2, 1)

s = left @ z @ right                       # (1, 1)

grad_table = backward(s)

print("s:")
print(s.data)

print("\ngrad s:")
print(grad_table[s])

print("\ngrad x:")
print(grad_table[x])

print("\ngrad a:")
print(grad_table[a])

print("\ngrad b:")
print(grad_table[b])
