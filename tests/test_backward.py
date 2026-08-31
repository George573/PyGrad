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
