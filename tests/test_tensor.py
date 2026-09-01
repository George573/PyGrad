import numpy as np

import pygrad.ops.operations as ops
from pygrad import Tensor


def test_tensor_exposes_array_metadata():
    tensor = Tensor(np.zeros((2, 3)))

    assert tensor.shape == (2, 3)
    assert tensor.ndim == 2
    assert tensor.size == 6
    assert tensor.device == "cpu"


def test_operation_connects_both_sides_of_computation_graph():
    left = Tensor(np.array([1.0, 2.0]))
    right = Tensor(np.array([3.0, 4.0]))

    result = left + right

    assert isinstance(result.op, ops.Add)
    assert result.inputs == (left, right)
    assert left.outputs == [result]
    assert right.outputs == [result]
    assert result.outputs == []


def test_operations_reject_tensors_on_different_devices():
    left = Tensor(1.0)
    right = Tensor(2.0)
    right.device = "cuda"

    with np.testing.assert_raises_regex(ValueError, "Cannot operate on tensor"):
        left + right
