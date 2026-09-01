import numpy as np
import pytest

import pygrad.ops.elementwise as ops
from pygrad import Tensor
from pygrad.optimizers.backprop import backward


@pytest.mark.parametrize(
    ("method", "values", "expected", "operation_type"),
    [
        ("exp", [-1.0, 0.0, 1.0], np.exp([-1.0, 0.0, 1.0]), ops.Exp),
        ("log", [0.5, 1.0, 2.0], np.log([0.5, 1.0, 2.0]), ops.Log),
        ("sqrt", [0.25, 1.0, 4.0], [0.5, 1.0, 2.0], ops.Sqrt),
        ("abs", [-2.0, 0.0, 3.0], [2.0, 0.0, 3.0], ops.Abs),
        ("tanh", [-1.0, 0.0, 1.0], np.tanh([-1.0, 0.0, 1.0]), ops.Tanh),
        ("sigmoid", [-1.0, 0.0, 1.0], [0.26894142, 0.5, 0.73105858], ops.Sigmoid),
        ("relu", [-2.0, 0.0, 3.0], [0.0, 0.0, 3.0], ops.ReLU),
    ],
)
def test_elementwise_forward(method, values, expected, operation_type):
    value = Tensor(np.array(values))

    result = getattr(value, method)()

    np.testing.assert_allclose(result.data, expected)
    assert isinstance(result.op, operation_type)


def test_abs_builtin_uses_elementwise_operation():
    value = Tensor(np.array([-2.0, 0.0, 3.0]))

    result = abs(value)

    np.testing.assert_allclose(result.data, [2.0, 0.0, 3.0])
    assert isinstance(result.op, ops.Abs)


@pytest.mark.parametrize(
    ("method", "value", "expected_gradient"),
    [
        ("exp", 1.0, np.e),
        ("log", 2.0, 0.5),
        ("sqrt", 4.0, 0.25),
        ("abs", -2.0, -1.0),
        ("tanh", 0.5, 1 - np.tanh(0.5) ** 2),
        ("sigmoid", 0.5, 0.2350037122015945),
        ("relu", 2.0, 1.0),
    ],
)
def test_elementwise_backward_supports_scalar_tensors(method, value, expected_gradient):
    input_tensor = Tensor(value, requires_grad=True)
    result = getattr(input_tensor, method)()

    gradients = backward(result)

    np.testing.assert_allclose(gradients[input_tensor], expected_gradient)
    assert gradients[input_tensor].shape == ()


@pytest.mark.parametrize(
    ("method", "expected_gradient"),
    [("abs", 0.0), ("relu", 0.0)],
)
def test_nondifferentiable_zero_uses_zero_subgradient(method, expected_gradient):
    value = Tensor(0.0, requires_grad=True)

    gradients = backward(getattr(value, method)())

    np.testing.assert_allclose(gradients[value], expected_gradient)


def test_sigmoid_is_stable_for_large_magnitudes():
    value = Tensor(np.array([-1000.0, 1000.0]), requires_grad=True)

    result = value.sigmoid()
    gradients = backward(result, np.ones_like(result.data))

    np.testing.assert_allclose(result.data, [0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(gradients[value], [0.0, 0.0], atol=1e-15)
