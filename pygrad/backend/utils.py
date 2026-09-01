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


def restore_reduced_dims(array, input_ndim, axis, keepdims):
    if axis is None or keepdims:
        return array

    if isinstance(axis, int):
        axis = (axis,)

    axis = tuple(ax if ax >= 0 else ax + input_ndim for ax in axis)

    for ax in sorted(axis):
        array = array.reshape(array.shape[:ax] + (1,) + array.shape[ax:])

    return array
