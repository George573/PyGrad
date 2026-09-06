# `Tensor`

```python
from pygrad import Tensor
```

```python
Tensor(data, device="cpu", requires_grad=False)
```

Wrap numerical data in a PyGrad tensor.

## Parameters

### `data`

A number, iterable of numbers, NumPy array, or CuPy array that can be converted
by the selected backend. Data is converted to `float32`.

### `device`

The storage and execution device:

- `"cpu"` uses NumPy and is the default;
- `"cuda"` uses CuPy and requires the `gpu` optional dependency.

Unknown device names raise `ValueError`. Requesting CUDA without CuPy installed
also raises `ValueError`.

### `requires_grad`

Whether operations involving this tensor should be recorded for reverse-mode
automatic differentiation. Defaults to `False`.

The implementation also accepts `op` and `inputs` arguments for tensors created
internally by operations. They are not intended for ordinary user code.

## Attributes and properties

### `data`

The underlying `numpy.ndarray` or `cupy.ndarray`. It is always `float32` and is
mutable.

### `device`

The string `"cpu"` or `"cuda"` supplied during construction.

### `requires_grad`

`True` when the tensor participates in gradient tracking. Operation results
inherit `True` when at least one input requires gradients.

### `grad`

The computed NumPy or CuPy gradient array, initially `None`. This attribute is
created only on tensors with `requires_grad=True`; `backward()` also assigns it
to its output tensor.

### `shape`

The tuple of array dimensions.

### `ndim`

The number of array dimensions.

### `size`

The total number of elements.

### `op`

The operation that created the tensor, or `None` for a leaf tensor.

### `inputs`

The creating operation's inputs that require gradients. Untracked operands are
not included.

## Arithmetic operators

| Syntax | Method | Description |
| --- | --- | --- |
| `a + b` | `a.__add__(b)` | Element-wise addition |
| `a - b` | `a.__sub__(b)` | Element-wise subtraction |
| `a * b` | `a.__mul__(b)` | Element-wise multiplication |
| `a / b` | `a.__truediv__(b)` | Element-wise division |
| `a**b` | `a.__pow__(b)` | Element-wise power |
| `a @ b` | `a.__matmul__(b)` | Matrix multiplication |
| `-a` | `a.__neg__()` | Element-wise negation |
| `abs(a)` | `a.__abs__()` | Element-wise absolute value |

Reverse forms are implemented for all binary operators, including
`__rmatmul__`. Binary operands may be tensors, numeric scalars, or NumPy/CuPy
arrays, although their shapes must be valid for the operation. Unsupported
operand types return `NotImplemented` through Python's operator protocol.

Tensor operands on different devices raise `ValueError`.

## Shape methods

### `reshape(shape)`

Return a tensor containing the same elements with a new shape.

- `shape`: an integer or tuple; other types raise `TypeError`.

```python
matrix = Tensor([1.0, 2.0, 3.0, 4.0]).reshape((2, 2))
```

### `flatten()`

Return a one-dimensional tensor containing the same elements.

### `transpose(axes=None)`

Transpose the tensor. With `axes=None`, reverse the axis order. Otherwise,
`axes` specifies the new order.

## Reduction methods

### `sum(axis=None, keepdims=False)`

Sum elements over all axes or selected axes.

- `axis`: `None`, an integer, or a tuple of integers;
- `keepdims`: retain reduced axes with size one when `True`.

### `mean(axis=None, keepdims=False)`

Calculate the arithmetic mean over all axes or selected axes. Parameters match
`sum()`.

## Element-wise methods

All methods return a new tensor and preserve gradient tracking.

| Method | Description |
| --- | --- |
| `exp()` | Exponential |
| `log()` | Natural logarithm |
| `sqrt()` | Square root |
| `abs()` | Absolute value; equivalent to `abs(tensor)` |
| `tanh()` | Hyperbolic tangent |
| `sigmoid()` | Numerically stable logistic sigmoid |
| `relu()` | Rectified linear unit |

Domain errors and non-finite values follow NumPy or CuPy floating-point
behavior.
