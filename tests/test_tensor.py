import numpy as np

import pygrad.ops.operations as ops
from pygrad import Tensor


def test_tensor_exposes_array_metadata():
    tensor = Tensor(np.zeros((2, 3)))

    assert tensor.shape == (2, 3)
    assert tensor.ndim == 2
    assert tensor.size == 6
    assert tensor.device == "cpu"


def test_operation_does_not_track_inputs_without_gradients():
    left = Tensor(np.array([1.0, 2.0]))
    right = Tensor(np.array([3.0, 4.0]))

    result = left + right

    assert isinstance(result.op, ops.Add)
    assert result.inputs == []
    assert result.requires_grad is False
    assert result.op.inputs == (left, right)


def test_operation_tracks_only_inputs_requiring_gradients():
    constant = Tensor(np.array([1.0, 2.0]))
    variable = Tensor(np.array([3.0, 4.0]), requires_grad=True)

    result = constant + variable

    assert result.inputs == (variable,)
    assert result.requires_grad is True
    assert result.grad is None


def test_requires_grad_is_inherited_through_operations():
    variable = Tensor(2.0, requires_grad=True)

    intermediate = variable * 3
    result = intermediate + Tensor(4.0)

    assert intermediate.requires_grad is True
    assert intermediate.inputs == (variable,)
    assert result.requires_grad is True
    assert result.inputs == (intermediate,)


def test_operations_reject_tensors_on_different_devices():
    left = Tensor(1.0)
    right = Tensor(2.0)
    right.device = "cuda"

    with np.testing.assert_raises_regex(ValueError, "Cannot operate on tensor"):
        left + right
