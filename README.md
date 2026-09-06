# PyGrad

PyGrad is a small automatic differentiation library implementing tensor
operations, dynamic computation graphs, and reverse-mode differentiation. It
uses NumPy on the CPU and can optionally use CuPy on a CUDA-capable GPU.

> [!NOTE]
> PyGrad is an early-stage educational project. Its API and supported operations
> are still growing.

## Features

- Dynamic computation graphs with selective gradient tracking
- Reverse-mode automatic differentiation
- NumPy broadcasting in forward and backward operations
- Element-wise arithmetic and matrix multiplication with NumPy/CuPy operands
- Reshaping, transposing, summation, and averaging
- Optional CuPy backend for CUDA devices
- A text representation of the computation graph

## Quick example

```python
from pygrad import Tensor
from pygrad.optimizers.backprop import backward

x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

# z = xy + x²
z = x * y + x**2
backward(z)

print(z.data) # [10.]
print(x.grad) # [7.]
print(y.grad) # [2.]
```

Tensors opt into gradient tracking with `requires_grad=True`. Operations record
only inputs that require gradients, and their results inherit gradient tracking.
Calling `backward()` follows this graph in reverse and stores each gradient in
the corresponding tensor's `.grad` attribute.

## Installation

PyGrad requires Python 3.10 or newer. To install the current source version:

```bash
git clone https://github.com/George573/PyGrad.git
cd PyGrad
python -m pip install -e .
```

For CUDA support, install the optional GPU dependencies instead:

```bash
python -m pip install -e ".[gpu]"
```

The GPU extra currently targets CUDA 12 through `cupy-cuda12x`. A compatible
NVIDIA driver and CUDA environment are required.

## Documentation

- [Getting started](docs/getting-started.md) — install PyGrad and calculate your
  first gradients
- [Tensors](docs/user-guide/tensors.md) — create tensors and work with their
  values and shapes
- [Automatic differentiation](docs/user-guide/autodiff.md) — understand graphs,
  backward passes, and gradient accumulation
- [Operations](docs/user-guide/operations.md) — arithmetic, element-wise
  functions, shape changes, and reductions
- [Devices](docs/user-guide/devices.md) — CPU/CUDA behavior and current data-type
  limitations
- [API reference](docs/api/index.md) — exact signatures, parameters, and return
  values

More runnable programs are available in [`usage_examples`](usage_examples/).

## Development

Install the development dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

Format the project with `make style`, or check formatting without changing
files with `make style-check`.

Build source and wheel distributions with:

```bash
python -m build
```

## License

PyGrad is distributed under the terms of the
[GNU General Public License v2.0 or later](LICENSE).
