# Devices

PyGrad uses NumPy for CPU tensors and optionally CuPy for CUDA tensors. The
chosen backend stores tensor data, evaluates operations, and stores gradients.

## CPU

CPU is the default device and requires no optional dependencies:

```python
from pygrad import Tensor

x = Tensor([1.0, 2.0, 3.0])

print(x.device) # cpu
print(type(x.data).__module__) # numpy
```

Passing `device="cpu"` explicitly has the same effect.

## CUDA

Install PyGrad's GPU extra in an environment with a compatible NVIDIA driver
and CUDA installation:

```bash
python -m pip install -e ".[gpu]"
```

The current extra installs `cupy-cuda12x`, which targets CUDA 12. Create GPU
tensors with `device="cuda"`:

```python
from pygrad import Tensor

x = Tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
y = (x * 2).sum()
```

If CuPy is unavailable, creating a CUDA tensor raises `ValueError` with an
installation hint.

## Data type

PyGrad currently converts all tensor data to 32-bit floating point:

| Device | Array type | Data type |
| --- | --- | --- |
| `"cpu"` | `numpy.ndarray` | `numpy.float32` |
| `"cuda"` | `cupy.ndarray` | `cupy.float32` |

This happens even when the constructor receives integers or another floating
type. Other tensor data types are not supported in this iteration.

## Keep operations on one device

Tensor operands in one operation must use the same device:

```python
cpu_value = Tensor([1.0], device="cpu")
cuda_value = Tensor([2.0], device="cuda")

# Raises ValueError:
result = cpu_value + cuda_value
```

PyGrad does not currently provide `.to()`, `.cpu()`, or `.cuda()` methods for
moving an existing tensor. Create a new tensor on the required device instead.

## Scalars and arrays

Numeric scalar operands are converted on the PyGrad tensor's device:

```python
cuda_value = Tensor([1.0, 2.0], device="cuda")
result = cuda_value * 2

print(result.device) # cuda
```

NumPy arrays belong with CPU tensors, while CuPy arrays belong with CUDA
tensors. Mixing an array from one backend with a tensor from the other is not
supported.

## Gradients stay on the device

Gradients use the same array backend as the computation:

```python
from pygrad.optimizers.backprop import backward

x = Tensor([1.0, 2.0], device="cuda", requires_grad=True)
loss = (x * x).sum()
backward(loss)

# x.grad is a cupy.ndarray
```

To inspect a CUDA result with NumPy, transfer it explicitly through CuPy:

```python
import cupy as cp

gradient_on_cpu = cp.asnumpy(x.grad)
```

The transfer creates an ordinary NumPy array outside PyGrad's computation graph.
