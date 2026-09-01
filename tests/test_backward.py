import numpy as np
import pytest

from pygrad import Tensor
from pygrad.optimizers.backprop import backward


def test_backward_accumulates_shared_graph_gradients():
    a = Tensor(np.array([[2.0]]))
    b = Tensor(np.array([[3.0]]))

    x = a * b
    y = x * a
    z = y + x

    gradients = backward(z)

    np.testing.assert_allclose(gradients[z], [[1.0]])
    np.testing.assert_allclose(gradients[y], [[1.0]])
    np.testing.assert_allclose(gradients[x], [[3.0]])
    np.testing.assert_allclose(gradients[a], [[15.0]])
    np.testing.assert_allclose(gradients[b], [[6.0]])


def test_backward_reduces_broadcast_gradients():
    column = Tensor(np.array([[1.0], [2.0], [3.0]]))
    matrix = Tensor(np.ones((3, 4)))
    result = column + matrix

    gradients = backward(result, np.ones_like(result.data))

    np.testing.assert_allclose(gradients[column], np.full((3, 1), 4.0))
    np.testing.assert_allclose(gradients[matrix], np.ones((3, 4)))


def test_backward_uses_explicit_upstream_gradient():
    value = Tensor(np.array([2.0, 3.0]))
    result = value * value

    gradients = backward(result, np.array([1.0, 2.0]))

    np.testing.assert_allclose(gradients[value], np.array([4.0, 12.0]))


def test_backward_matrix_multiplication_gradients():
    left = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    right = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    result = left @ right
    upstream = np.array([[1.0, 2.0], [3.0, 4.0]])

    gradients = backward(result, upstream)

    np.testing.assert_allclose(gradients[left], upstream @ right.data.T)
    np.testing.assert_allclose(gradients[right], left.data.T @ upstream)


@pytest.mark.parametrize(
    ("left_data", "right_data"),
    [
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ([1.0, 2.0], [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
        ([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0]),
    ],
)
def test_backward_matrix_multiplication_with_vectors(left_data, right_data):
    left = Tensor(np.array(left_data))
    right = Tensor(np.array(right_data))
    result = left @ right
    upstream = np.ones_like(result.data)

    gradients = backward(result, upstream)

    def objective(left_value, right_value):
        return np.sum((left_value @ right_value) * upstream)

    epsilon = 1e-6
    for tensor, other, gradient, is_left in (
        (left, right, gradients[left], True),
        (right, left, gradients[right], False),
    ):
        expected = np.empty_like(tensor.data)
        for index in np.ndindex(tensor.shape):
            plus = tensor.data.copy()
            minus = tensor.data.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            args_plus = (plus, other.data) if is_left else (other.data, plus)
            args_minus = (minus, other.data) if is_left else (other.data, minus)
            expected[index] = (objective(*args_plus) - objective(*args_minus)) / (
                2 * epsilon
            )
        np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-6)


def test_backward_division_gradients_with_broadcasting():
    numerator = Tensor(np.array([[2.0], [4.0]]))
    denominator = Tensor(np.array([[1.0, 2.0, 4.0]]))
    result = numerator / denominator
    upstream = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    gradients = backward(result, upstream)

    expected_numerator = (upstream / denominator.data).sum(axis=1, keepdims=True)
    expected_denominator = (-upstream * numerator.data / denominator.data**2).sum(
        axis=0, keepdims=True
    )
    np.testing.assert_allclose(gradients[numerator], expected_numerator)
    np.testing.assert_allclose(gradients[denominator], expected_denominator)


def test_backward_negation_gradients():
    value = Tensor(np.array([1.0, -2.0, 3.0]))
    upstream = np.array([2.0, 4.0, 6.0])

    gradients = backward(-value, upstream)

    np.testing.assert_allclose(gradients[value], -upstream)


def test_backward_power_gradients_with_broadcasting():
    base = Tensor(np.array([[2.0], [3.0]]))
    exponent = Tensor(np.array([[2.0, 3.0]]))
    result = base**exponent
    upstream = np.array([[1.0, 2.0], [3.0, 4.0]])

    gradients = backward(result, upstream)

    expected_base = (upstream * exponent.data * base.data ** (exponent.data - 1)).sum(
        axis=1, keepdims=True
    )
    expected_exponent = (upstream * result.data * np.log(base.data)).sum(
        axis=0, keepdims=True
    )
    np.testing.assert_allclose(gradients[base], expected_base)
    np.testing.assert_allclose(gradients[exponent], expected_exponent)


@pytest.mark.parametrize("method", ["reshape", "flatten"])
def test_backward_shape_operation_gradients(method):
    value = Tensor(np.arange(6.0).reshape(2, 3))
    result = value.reshape((3, 2)) if method == "reshape" else value.flatten()
    upstream = np.arange(1.0, 7.0).reshape(result.shape)

    gradients = backward(result, upstream)

    np.testing.assert_allclose(gradients[value], upstream.reshape(value.shape))


@pytest.mark.parametrize("axes", [None, (2, 0, 1)])
def test_backward_transpose_gradients(axes):
    value = Tensor(np.arange(24.0).reshape(2, 3, 4))
    result = value.transpose(axes)
    upstream = np.arange(24.0).reshape(result.shape)

    gradients = backward(result, upstream)

    if axes is None:
        expected = upstream.transpose()
    else:
        inverse_axes = np.argsort(axes)
        expected = upstream.transpose(inverse_axes)
    np.testing.assert_allclose(gradients[value], expected)


@pytest.mark.parametrize(
    ("method", "axis", "keepdims", "scale"),
    [
        ("sum", None, False, 1.0),
        ("sum", -1, False, 1.0),
        ("sum", (0, 2), False, 1.0),
        ("sum", (0, 2), True, 1.0),
        ("mean", None, False, 24.0),
        ("mean", -1, False, 4.0),
        ("mean", (0, 2), False, 8.0),
        ("mean", (0, 2), True, 8.0),
    ],
)
def test_backward_reduction_gradients(method, axis, keepdims, scale):
    value = Tensor(np.arange(24.0).reshape(2, 3, 4))
    result = getattr(value, method)(axis=axis, keepdims=keepdims)
    upstream = np.arange(1.0, result.size + 1.0).reshape(result.shape)

    gradients = backward(result, upstream)

    expanded = upstream
    if axis is not None and not keepdims:
        axes = (axis,) if isinstance(axis, int) else axis
        expanded = np.expand_dims(upstream, axis=axes)
    expected = np.broadcast_to(expanded / scale, value.shape)
    np.testing.assert_allclose(gradients[value], expected)


def test_backward_requires_gradient_for_multi_element_output():
    value = Tensor(np.array([2.0, 3.0]))
    result = value * value

    with pytest.raises(RuntimeError, match="gradient must be provided"):
        backward(result)


def test_backward_rejects_gradient_with_wrong_shape():
    value = Tensor(np.array([2.0, 3.0]))
    result = value * value

    with pytest.raises(ValueError, match="gradient shape .* output shape"):
        backward(result, np.array([1.0]))
