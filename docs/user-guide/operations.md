# Operations

PyGrad operations calculate values immediately and, when needed, connect their
results to the dynamic computation graph. A result requires gradients when at
least one of its tensor inputs has `requires_grad=True`.

## Operands

Binary operators accept PyGrad tensors, numeric scalars, and NumPy or CuPy
arrays. Non-tensor operands are converted to untracked tensors on the same
device as the PyGrad operand.

```python
import numpy as np

from pygrad import Tensor

x = Tensor([1.0, 2.0], requires_grad=True)

x + 2.0
x * np.array([3.0, 4.0])
np.array([3.0, 4.0]) @ x
```

Python lists are accepted by the `Tensor` constructor, but are not converted
automatically when used directly as binary operands. Wrap them in `Tensor` or a
NumPy/CuPy array first.

## Arithmetic

| Expression | Operation | Notes |
| --- | --- | --- |
| `a + b` | Addition | Element-wise |
| `a - b` | Subtraction | Element-wise |
| `a * b` | Multiplication | Element-wise |
| `a / b` | Division | Element-wise; division by zero follows backend behavior |
| `-a` | Negation | Element-wise |
| `a**b` | Power | Element-wise |
| `a @ b` | Matrix multiplication | Uses NumPy/CuPy matmul shape rules |

Binary arithmetic works in both operand orders:

```python
x = Tensor([2.0, 4.0], requires_grad=True)

x + 1
1 + x
x - 1
1 - x
x * 2
2 * x
x / 2
2 / x
x**2
2**x
```

### Broadcasting

Element-wise operations follow NumPy or CuPy broadcasting rules:

```python
matrix = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
bias = Tensor([0.5, -0.5], requires_grad=True)
result = matrix + bias
```

During backpropagation, PyGrad sums over broadcast dimensions so each input
gradient has the same shape as its tensor.

### Matrix multiplication

The `@` operator supports vectors, matrices, and higher-dimensional batches as
allowed by the active array backend. NumPy/CuPy arrays work on either side:

```python
import numpy as np

from pygrad import Tensor

weights = Tensor([[1.0], [2.0]], requires_grad=True)

first = np.array([[3.0, 4.0]]) @ weights
second = weights.transpose() @ np.array([[3.0], [4.0]])
```

An array converted during the expression is an untracked constant. Construct a
`Tensor(..., requires_grad=True)` explicitly if its gradient is needed.

## Element-wise functions

| Expression | Function | Derivative |
| --- | --- | --- |
| `-x` | Negation | `-1` |
| `abs(x)` or `x.abs()` | Absolute value | `sign(x)`; zero at `x = 0` |
| `x.exp()` | Exponential | `exp(x)` |
| `x.log()` | Natural logarithm | `1 / x` |
| `x.sqrt()` | Square root | `1 / (2 * sqrt(x))` |
| `x.tanh()` | Hyperbolic tangent | `1 - tanh(x)²` |
| `x.sigmoid()` | Logistic sigmoid | `sigmoid(x) * (1 - sigmoid(x))` |
| `x.relu()` | Rectified linear unit | `1` above zero; `0` otherwise |

```python
x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
y = x.sigmoid() + x.relu()
```

Backend floating-point rules apply. For real-valued calculations, `log()`
expects positive inputs and `sqrt()` expects non-negative inputs. The power
gradient with respect to an exponent uses `log(base)`, so differentiating that
operand normally requires a positive base.

## Shape operations

### `reshape(shape)`

Changes the shape without changing the number of elements:

```python
x = Tensor([1.0, 2.0, 3.0, 4.0])
matrix = x.reshape((2, 2))
```

### `flatten()`

Returns a one-dimensional tensor:

```python
vector = matrix.flatten()
```

### `transpose(axes=None)`

Reverses the axes by default or applies an explicit axis order:

```python
x = Tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
reversed_axes = x.transpose()
reordered = x.transpose((2, 0, 1))
```

All shape operations preserve gradient tracking.

## Reductions

`sum()` and `mean()` reduce all elements by default:

```python
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

total = x.sum()
average = x.mean()
```

Use `axis` to reduce selected dimensions and `keepdims=True` to retain them:

```python
columns = x.sum(axis=0)
rows = x.mean(axis=1, keepdims=True)
```

`axis` may be `None`, an integer, or a tuple of integers. Reductions are the
usual way to turn a multi-element result into a one-element loss before calling
`backward()`.

## Combining operations

Operations can be freely composed:

```python
from pygrad.optimizers.backprop import backward

features = Tensor([[1.0, -2.0]], requires_grad=True)
weights = Tensor([[0.5], [1.5]], requires_grad=True)

prediction = (features @ weights).sigmoid()
error = prediction - 1.0
loss = (error * error).mean()
backward(loss)
```

See [Automatic differentiation](autodiff.md) for gradient seeding and graph
traversal.
