try:
    import cupy as cp
except ImportError:
    cp = None  # CuPy is optional; handle the case where it's not installed


def unbroadcast(tensor, shape):
    if tensor.shape == shape:
        return tensor

    extra_dims = tensor.ndim - len(shape)

    axes = tuple(range(extra_dims))
    axes += tuple(i + extra_dims for i, size in enumerate(shape) if size == 1)

    if axes:
        tensor = tensor.sum(axis=axes, keepdims=True)

    return tensor.reshape(shape)
