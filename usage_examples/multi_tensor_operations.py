"""Build a larger graph and calculate gradients for several input tensors."""

import numpy as np

from pygrad import Tensor
from pygrad.optimizers.backprop import backward

# Five independent inputs with different shapes. Broadcasting allows bias and
# scale to participate in the same graph as the matrices.
features = Tensor(
    np.array(
        [
            [1.0, -2.0, 0.5],
            [0.0, 3.0, -1.0],
        ]
    ),
    device="cuda",
    requires_grad=True,
)
weights = Tensor(
    np.array(
        [
            [0.5, -1.0],
            [1.5, 0.25],
            [-0.5, 2.0],
        ]
    ),
    device="cuda",
    requires_grad=True,
)
bias = Tensor(np.array([[0.1, -0.2]]), device="cuda", requires_grad=True)
scale = Tensor(np.array(2.0), device="cuda", requires_grad=True)
exponent = Tensor(np.array(2.0), device="cuda", requires_grad=True)
one = Tensor(np.array(1.0), device="cuda")

# Matrix multiplication, broadcasting, and several element-wise branches.
linear = features @ weights + bias
smooth_branch = linear.tanh() + linear.sigmoid()
positive_branch = (linear.abs() + one).sqrt().log().exp()
relu_branch = linear.relu()

# Merge the branches, apply binary element-wise operations, and change shape.
combined = (smooth_branch + relu_branch) * positive_branch / scale
powered = combined.abs() ** exponent
reordered = powered.reshape((2, 2)).transpose((1, 0)).flatten()

# A scalar output lets backward() create the initial gradient automatically.
loss = -reordered.mean()
backward(loss)

print("forward operation outputs:")
for name, tensor in (
    ("linear = features @ weights + bias", linear),
    ("smooth_branch = tanh(linear) + sigmoid(linear)", smooth_branch),
    ("positive_branch = exp(log(sqrt(abs(linear) + 1)))", positive_branch),
    ("relu_branch = relu(linear)", relu_branch),
    ("combined = (smooth + relu) * positive / scale", combined),
    ("powered = abs(combined) ** exponent", powered),
    ("reordered = reshape -> transpose -> flatten", reordered),
    ("loss = -mean(reordered)", loss),
):
    print(f"\n{name}:")
    print(tensor.data)

print("\ninput gradients:")
for name, tensor in (
    ("features", features),
    ("weights", weights),
    ("bias", bias),
    ("scale", scale),
    ("exponent", exponent),
):
    print(f"\ngradient for {name}:")
    print(tensor.grad)
