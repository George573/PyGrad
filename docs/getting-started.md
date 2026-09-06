# Getting started

This tutorial takes you from installing PyGrad to calculating derivatives with
a small computation graph. No knowledge of PyGrad's internals is required.

## Requirements

PyGrad requires Python 3.10 or newer. NumPy is installed automatically as its
CPU array backend.

## Install PyGrad

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/George573/PyGrad.git
cd PyGrad
python -m pip install -e .
```

Editable mode means changes to the checked-out source are immediately available
to Python without reinstalling the package.

Confirm that the installation works:

```bash
python -c "import pygrad; print(pygrad.__version__)"
```

To contribute to PyGrad, include its testing and formatting tools:

```bash
python -m pip install -e ".[dev]"
```

## Create tensors

A `Tensor` wraps numerical data held by the selected array backend. Python
lists, numbers, and NumPy arrays can be passed to its constructor. Set
`requires_grad=True` on values whose gradients you want to calculate:

```python
import numpy as np

from pygrad import Tensor

scalar = Tensor(2.0)
vector = Tensor([1.0, 2.0, 3.0], requires_grad=True)
matrix = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

print(vector.data)  # [1. 2. 3.]
print(matrix.shape) # (2, 2)
```

> **Current limitation:** PyGrad converts all tensor data to `float32`, even
> when the input uses another numeric data type. Other tensor data types are not
> supported in this iteration.

## Build a computation

PyGrad supports tensor arithmetic through familiar Python operators. In this
example, the expression is `loss = xy + x²`:

```python
from pygrad import Tensor

x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

product = x * y
square = x**2
loss = product + square

print(loss.data) # [10.]
```

These statements calculate the result immediately. Because `x` and `y` require
gradients, their downstream results inherit `requires_grad=True`. Each result
records only the inputs that require gradients.

Binary operations use the same operand conversion and accept tensors, numeric
scalars, and NumPy or CuPy arrays. The operands must still have shapes that are
valid for the chosen operation; in particular, matrix multiplication requires
arrays with suitable dimensions.

NumPy and CuPy arrays work on either side of matrix multiplication:

```python
import numpy as np

weights = Tensor([[1.0], [2.0]], requires_grad=True)

left_result = np.array([[3.0, 4.0]]) @ weights
right_result = weights.transpose() @ np.array([[3.0], [4.0]])
```

An array operand is converted to an untracked tensor on the same device as the
PyGrad tensor. Consequently, `backward()` calculates a gradient for `weights`,
but not for the original NumPy arrays.

## Calculate gradients

Import `backward` and pass it the final output tensor. Without a second
argument, that output must contain exactly one element. This gives PyGrad one
starting value whose gradient can be initialized to `1`, allowing the backward
pass to proceed through the graph:

```python
from pygrad.optimizers.backprop import backward

backward(loss)

print(x.grad) # [7.]
print(y.grad) # [2.]
```

For `loss = xy + x²`:

- the derivative with respect to `x` is `y + 2x`, which is `7`;
- the derivative with respect to `y` is `x`, which is `2`.

`backward()` stores a NumPy or CuPy array in the `.grad` attribute of every
gradient-tracked tensor it reaches. Tensors created with `requires_grad=False`
do not have a `.grad` attribute.

The single-element rule concerns the total number of elements, not the number
of dimensions. Outputs with shapes `()`, `(1,)`, or `(1, 1)` all contain one
element and can be passed to `backward()` without an explicit upstream
gradient. For an output with multiple elements, either reduce it to one element
or provide an upstream gradient.

## Handle multi-element results

### Reduce the output

Calling `backward()` without an explicit starting gradient requires the output
to contain exactly one element. A common way to produce such an output is a
reduction:

```python
from pygrad import Tensor
from pygrad.optimizers.backprop import backward

values = Tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (values * values).sum()
backward(loss)

print(loss.data)   # 14.0
print(values.grad) # [2. 4. 6.]
```

### Provide an upstream gradient

For a multi-element output, supply an upstream gradient with exactly the same
shape:

```python
import numpy as np

values = Tensor([2.0, 3.0], requires_grad=True)
result = values * values
backward(result, np.array([1.0, 2.0]))

print(values.grad) # [ 4. 12.]
```

## Use a CUDA device

Install the optional backend:

```bash
python -m pip install -e ".[gpu]"
```

Then create tensors on CUDA explicitly:

```python
from pygrad import Tensor

x = Tensor([1.0, 2.0], device="cuda", requires_grad=True)
y = Tensor([3.0, 4.0], device="cuda", requires_grad=True)
result = x + y
```

All tensors in one operation must use the same device. PyGrad does not currently
provide a method that moves an existing tensor between CPU and CUDA.

## Where to go next

- Read the [tensor guide](user-guide/tensors.md) for supported operations and
  shape transformations.
- Read [automatic differentiation](user-guide/autodiff.md) for computation
  graphs, upstream gradients, and accumulation.
- Try the programs in [`usage_examples`](../usage_examples/).
