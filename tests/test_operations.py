import operator

import numpy as np
import pytest

import pygrad.ops.operations as ops
from pygrad import Tensor


@pytest.mark.parametrize(
    ("operation", "operation_type", "expected"),
    [
        (operator.add, ops.Add, [5.0, 7.0, 9.0]),
        (operator.sub, ops.Sub, [-3.0, -3.0, -3.0]),
        (operator.mul, ops.Mul, [4.0, 10.0, 18.0]),
        (operator.truediv, ops.Div, [0.25, 0.4, 0.5]),
        (operator.pow, ops.Pow, [1.0, 32.0, 729.0]),
    ],
)
def test_elementwise_binary_operation_values(operation, operation_type, expected):
    left = Tensor(np.array([1.0, 2.0, 3.0]))
    right = Tensor(np.array([4.0, 5.0, 6.0]))

    result = operation(left, right)

    np.testing.assert_allclose(result.data, expected)
    assert isinstance(result.op, operation_type)


def test_negation_values():
    value = Tensor(np.array([1.0, -2.0, 3.0]))

    result = -value

    np.testing.assert_allclose(result.data, [-1.0, 2.0, -3.0])
    assert isinstance(result.op, ops.Neg)


def test_addition_broadcasts_values():
    column = Tensor(np.array([[1.0], [2.0], [3.0]]))
    matrix = Tensor(np.ones((3, 4)))

    result = column + matrix

    np.testing.assert_allclose(
        result.data,
        [
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 4.0],
        ],
    )
    assert result.shape == (3, 4)


def test_matrix_multiplication_values():
    left = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    right = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    result = left @ right

    np.testing.assert_allclose(result.data, [[22.0, 28.0], [49.0, 64.0]])
    assert isinstance(result.op, ops.MatMul)
    assert result.shape == (2, 2)


@pytest.mark.parametrize(
    ("method", "kwargs", "expected", "operation_type"),
    [
        ("sum", {}, 21.0, ops.Sum),
        ("sum", {"axis": 0}, [5.0, 7.0, 9.0], ops.Sum),
        ("mean", {}, 3.5, ops.Mean),
        ("mean", {"axis": 1, "keepdims": True}, [[2.0], [5.0]], ops.Mean),
    ],
)
def test_reduction_values(method, kwargs, expected, operation_type):
    value = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    result = getattr(value, method)(**kwargs)

    np.testing.assert_allclose(result.data, expected)
    assert isinstance(result.op, operation_type)


def test_reshape_and_flatten_values():
    value = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))

    reshaped = value.reshape((2, 2))
    flattened = reshaped.flatten()

    np.testing.assert_allclose(reshaped.data, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(flattened.data, [1.0, 2.0, 3.0, 4.0])
    assert isinstance(reshaped.op, ops.Reshape)
    assert isinstance(flattened.op, ops.Flatten)


def test_transpose_values_with_explicit_axes():
    value = Tensor(np.arange(24.0).reshape(2, 3, 4))

    result = value.transpose((2, 0, 1))

    np.testing.assert_allclose(result.data, value.data.transpose((2, 0, 1)))
    assert isinstance(result.op, ops.Transpose)
    assert result.shape == (4, 2, 3)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda value: value + 2, 6.0),
        (lambda value: 2 + value, 6.0),
        (lambda value: value - 2, 2.0),
        (lambda value: 2 - value, -2.0),
        (lambda value: value * 2, 8.0),
        (lambda value: 2 * value, 8.0),
        (lambda value: value / 2, 2.0),
        (lambda value: 2 / value, 0.5),
        (lambda value: value**2, 16.0),
        (lambda value: 2**value, 16.0),
    ],
)
def test_binary_operations_accept_scalars_on_either_side(operation, expected):
    result = operation(Tensor(4.0))

    np.testing.assert_allclose(result.data, expected)
    assert result.device == "cpu"


def test_compatibility_exports_are_canonical_classes():
    from pygrad.ops.arithmetic import Add
    from pygrad.ops.elementwise import Neg
    from pygrad.ops.reductions import Sum
    from pygrad.ops.shape import Reshape

    assert ops.Add is Add
    assert ops.Neg is Neg
    assert ops.Sum is Sum
    assert ops.Reshape is Reshape
