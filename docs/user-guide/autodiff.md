# Automatic differentiation

PyGrad calculates derivatives with reverse-mode automatic differentiation.
Tensors opt into tracking with `requires_grad=True`; operations retain those
inputs, and their downstream results inherit tracking. PyGrad then walks that
dynamic graph backward from a chosen output. Untracked inputs are not added to
the backward graph.

## The computation graph

Consider this expression:

```python
from pygrad import Tensor

a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
x = a * b
result = x + x
```

When this path requires gradients, the graph records that `x` was produced from
`a` and `b`, and that `result` uses `x` twice. The graph is dynamic: ordinary
Python execution determines its structure each time the program runs.

## Run a backward pass

Call `backward()` with the tensor whose derivative should seed the traversal:

```python
from pygrad.optimizers.backprop import backward

backward(result)

print(a.grad) # [6.]
print(b.grad) # [4.]
```

`backward()` returns `None`. It stores NumPy or CuPy gradient arrays in the
`.grad` attributes of the output, tracked intermediates, and tracked leaf
tensors reached during traversal. Untracked constants are excluded from the
backward graph and do not have `.grad`.

## Scalar and multi-element outputs

The simplest approach is to reduce a multi-element result to one value with
`sum()` or `mean()`. This is common when a model produces a scalar loss:

```python
x = Tensor([2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
backward(loss)
```

Because `loss` contains one element, `backward()` automatically seeds it with
`1`. Reductions remain part of the computation graph, so the resulting gradient
is propagated back to every contributing element of `x`.

If you do not reduce an output, PyGrad cannot assume which vector-Jacobian
product you want. Pass an upstream gradient with the same shape as the output:

```python
import numpy as np

x = Tensor([2.0, 3.0], requires_grad=True)
y = x * x

backward(y, np.array([1.0, 2.0]))
print(x.grad) # [ 4. 12.]
```

Calling `backward(y)` without that argument raises `RuntimeError`. Passing a
gradient with a different shape raises `ValueError`.

### Differentiate one output element

An upstream gradient containing one `1` and zeros everywhere else selects a
single element of a multi-element output. This calculates the gradient of that
selected output element with respect to the tracked inputs:

```python
import numpy as np

x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = x * x

upstream = np.zeros_like(y.data)
upstream[1, 1] = 1.0
backward(y, upstream)

print(x.grad)
# [[0. 0.]
#  [0. 8.]]
```

Here the `1` selects `y[1, 1]`, so PyGrad computes the derivative of that value
with respect to all elements of `x`. Only `x[1, 1]` has a nonzero gradient in
this example because `y = x * x` is element-wise. For operations that mix
values, such as matrix multiplication, one selected output can depend on several
input elements and therefore produce several nonzero gradients.

## Shared computations accumulate gradients

A tensor may contribute to an output through several paths. PyGrad waits for
all downstream contributions and adds them before continuing backward:

```python
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)

x = a * b
y = x * a
z = y + x

backward(z)
print(x.grad) # [3.]
```

Here, `x` reaches `z` directly and through `y`, so its contributions are
accumulated rather than overwritten.

## Broadcasting in the backward pass

Forward element-wise operations follow NumPy or CuPy broadcasting. During the
backward pass, PyGrad sums expanded dimensions so that each gradient matches its
tensor's original shape:

```python
import numpy as np

column = Tensor([[1.0], [2.0], [3.0]], requires_grad=True) # (3, 1)
matrix = Tensor(np.ones((3, 4)), requires_grad=True)       # (3, 4)
result = column + matrix

backward(result, np.ones_like(result.data))
print(column.grad.shape) # (3, 1)
print(matrix.grad.shape) # (3, 4)
```

## Supported differentiable operations

Backward implementations currently exist for:

- addition, subtraction, multiplication, division, negation, and absolute value;
- exponential, logarithm, square root, tanh, sigmoid, and ReLU;
- powers;
- matrix multiplication, including vector inputs;
- reshape, flatten, and transpose;
- sum and mean reductions;
- broadcasting used by element-wise operations.

The exponent gradient for `a**b` uses `log(a)`, so differentiating with respect
to a real-valued exponent requires positive base values for ordinary real-number
results.

Binary operations, including matrix multiplication, may combine tensors with
NumPy or CuPy arrays. PyGrad converts the array operand to an untracked tensor,
so gradients are calculated only for explicit tensors created with
`requires_grad=True`.

## Inspect a graph

PyGrad can print the graph rooted at an output:

```python
from pygrad.utils.draw import print_graph

print_graph(result)
```

The output includes tensor values, object addresses, operations, and their input
tensors. Object addresses vary from run to run.

## Current model

PyGrad's automatic differentiation is intentionally small and explicit:

- gradients are stored in each tracked tensor's `.grad` attribute;
- tracking begins at tensors with `requires_grad=True` and propagates through
  their downstream results;
- the graph is retained after a backward pass;
- there is currently no public graph-detachment operation;
- higher-order differentiation is not currently supported because gradients
  are NumPy or CuPy arrays rather than tensors.

Return to [Getting started](../getting-started.md) or continue exploring the
[Tensor guide](tensors.md).
