# Tensors

`Tensor` is PyGrad's central data container. It stores numerical values on a
CPU or CUDA device and optionally participates in automatic differentiation.
Calculations performed with tensors are covered separately in the
[Operations guide](operations.md).

## Construct a tensor

Import `Tensor` from the top-level package:

```python
from pygrad import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
```

The user-facing constructor arguments are:

```python
Tensor(data, device="cpu", requires_grad=False)
```

- `data` is a number, iterable, NumPy array, or CuPy array understood by the
  selected backend.
- `device` selects `"cpu"` or `"cuda"`; CPU is the default.
- `requires_grad` controls whether PyGrad tracks this tensor for automatic
  differentiation; it is `False` by default.

The constructor also has `op` and `inputs` arguments used internally when an
operation creates a result. User code should not set them directly.

## Data storage

Tensor values are stored in `.data` as an array:

```python
x = Tensor([[1, 2], [3, 4]])

print(x.data)
# [[1. 2.]
#  [3. 4.]]
```

PyGrad currently converts all inputs to `float32`. Other tensor data types are
not supported in this iteration.

- CPU tensors store a `numpy.ndarray`.
- CUDA tensors store a `cupy.ndarray`.

See [Devices](devices.md) for installation, backend matching, and gradient
placement.

## Inspect a tensor

Properties expose the underlying array's dimensions:

```python
x = Tensor([[1.0, 2.0], [3.0, 4.0]])

print(x.shape)  # (2, 2)
print(x.ndim)   # 2
print(x.size)   # 4
print(x.device) # cpu
```

| Attribute | Meaning |
| --- | --- |
| `.data` | Underlying NumPy or CuPy array |
| `.device` | `"cpu"` or `"cuda"` |
| `.shape` | Tuple of dimension sizes |
| `.ndim` | Number of dimensions |
| `.size` | Total number of elements |
| `.requires_grad` | Whether the tensor participates in gradient tracking |
| `.grad` | Computed gradient, when gradient tracking is enabled |

Printing a tensor shows its stored values. Its developer-oriented
representation also includes an object address, which can help distinguish
tensors while inspecting a graph.

## Constants and trainable values

A tensor is an untracked constant by default:

```python
constant = Tensor([1.0, 2.0])

print(constant.requires_grad) # False
print(hasattr(constant, "grad")) # False
```

Set `requires_grad=True` for a value whose gradient you need:

```python
weight = Tensor([1.0, 2.0], requires_grad=True)

print(weight.requires_grad) # True
print(weight.grad)          # None
```

The `.grad` attribute starts as `None` and receives a NumPy or CuPy array after
`backward()` reaches the tensor.

Choose `requires_grad` when constructing a tensor. Changing it later is not a
documented workflow because the surrounding computation graph may already have
been built.

## Tracking propagates to results

When an operation uses at least one tracked tensor, its result also requires
gradients:

```python
weight = Tensor([2.0], requires_grad=True)
bias = Tensor([1.0])
result = weight * 3 + bias

print(result.requires_grad) # True
```

Untracked operands act as constants. In this example, PyGrad retains the path
to `weight` but does not calculate a gradient for `bias`.

If no input requires gradients, the result is untracked:

```python
a = Tensor([1.0])
b = Tensor([2.0])
result = a + b

print(result.requires_grad) # False
```

## Gradient state

After a backward pass, gradients live on the tensor:

```python
from pygrad.optimizers.backprop import backward

x = Tensor([2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
backward(loss)

print(x.grad) # [4. 6.]
```

PyGrad does not clear existing input gradients automatically. When training
with `SGD`, call `optimizer.zero_grad()` before the next backward pass. See
[Automatic differentiation](autodiff.md) for scalar outputs, explicit upstream
gradients, and accumulation.

## Graph metadata

Two attributes describe how a tensor was created:

- `.op` is the operation object that produced the tensor, or `None` for a leaf;
- `.inputs` contains the producing operation's inputs that require gradients.

For example:

```python
constant = Tensor([1.0])
variable = Tensor([2.0], requires_grad=True)
result = constant + variable

print(constant.op)       # None
print(result.op)         # Add
print(result.inputs)     # contains only variable
print(result.op.inputs)  # contains constant and variable
```

`result.op.inputs` contains all operands needed by the operation's derivative,
whereas `result.inputs` contains only the paths traversed during
backpropagation. These attributes are useful for inspection but normally should
not be modified directly.

Use the [graph utilities](../api/utilities.md) to display these connections.
