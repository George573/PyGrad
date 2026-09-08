import operator

import numpy as np
import pytest

from pygrad import Tensor
from pygrad.ops.arithmetic import Mul
from pygrad.ops.ops import Ops
from pygrad.optimizers.backprop import backward
from pygrad.optimizers.sgd import SGD


@pytest.mark.parametrize("clear_leaf", [False, True])
def test_repeated_backward_uses_fresh_intermediate_gradients(clear_leaf):
    x = Tensor(2.0, requires_grad=True)
    intermediate = x * x
    loss = intermediate * 3
    backward(loss)
    np.testing.assert_allclose(x.grad, 12)
    if clear_leaf:
        SGD([x]).zero_grad()
    backward(loss)
    np.testing.assert_allclose(x.grad, 12 if clear_leaf else 24)
    np.testing.assert_allclose(intermediate.grad, 6)
    np.testing.assert_allclose(loss.grad, 2)


def test_separate_losses_share_an_intermediate():
    x = Tensor(2.0, requires_grad=True)
    intermediate = x * x
    first = intermediate * 3
    second = intermediate * 5
    backward(first)
    backward(second)
    np.testing.assert_allclose(x.grad, 32)
    np.testing.assert_allclose(intermediate.grad, 8)


def test_backward_on_leaf_accumulates_different_seeds():
    x = Tensor(2.0, requires_grad=True)
    backward(x)
    np.testing.assert_allclose(x.grad, 1)
    backward(x, np.array(3.0))
    np.testing.assert_allclose(x.grad, 4)


@pytest.mark.parametrize("kind", ["missing", "extra", "wrong_shape"])
def test_backward_rejects_malformed_operation_gradients(kind):
    class BrokenOperation(Ops):
        def backward(self, grad):
            if kind == "missing":
                return ()
            if kind == "extra":
                return (np.ones(2), np.ones(2))
            return (np.ones(3),)

    x = Tensor([1.0, 2.0], requires_grad=True)
    op = BrokenOperation(x)
    output = op.create_tensor(np.array(1.0), op, (x,))
    message = "Gradient shape mismatch" if kind == "wrong_shape" else "input gradients"
    with pytest.raises(RuntimeError, match=message):
        backward(output)
    assert x.grad is None


def test_rejected_operation_reuse_preserves_original_graph():
    x = Tensor(2.0, requires_grad=True)
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(5.0, requires_grad=True)
    op = Mul()
    output = op(x, a)
    with pytest.raises(RuntimeError, match="only be called once"):
        op(x, b)
    backward(output)
    np.testing.assert_allclose(x.grad, 3)
    np.testing.assert_allclose(a.grad, 2)
    assert b.grad is None


def test_untracked_chain_has_no_backward_history():
    value = Tensor(1.0)
    for _ in range(20):
        value = value * 2 + 1
        assert value.op is None
        assert not value.inputs
        assert not value.requires_grad


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((2, 1, 3, 4), (1, 5, 4, 2)),
        ((4,), (2, 4, 3)),
        ((2, 3, 4), (4,)),
    ],
)
def test_batched_matmul_against_numerical_reference(
    left_shape, right_shape, numerical_gradients
):
    rng = np.random.default_rng(42)
    left = Tensor(rng.normal(size=left_shape), requires_grad=True)
    right = Tensor(rng.normal(size=right_shape), requires_grad=True)
    output = left @ right
    upstream = rng.normal(size=output.shape).astype(np.float32)
    backward(output, upstream)
    expected = numerical_gradients(
        lambda a, b: np.sum((a @ b) * upstream), [left.data, right.data]
    )
    for tensor, gradient in zip((left, right), expected):
        assert tensor.grad.shape == tensor.shape
        np.testing.assert_allclose(tensor.grad, gradient, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "operation",
    [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow],
    ids=["add", "sub", "mul", "div", "pow"],
)
@pytest.mark.parametrize(
    "left_shape,right_shape",
    [((), (2, 3)), ((3,), (2, 3)), ((2, 1, 3, 1), (1, 4, 1, 2))],
)
def test_broadcast_gradients_against_numerical_reference(
    operation, left_shape, right_shape, numerical_gradients
):
    rng = np.random.default_rng(10)
    left = Tensor(rng.uniform(0.5, 2, size=left_shape), requires_grad=True)
    right = Tensor(rng.uniform(0.5, 2, size=right_shape), requires_grad=True)
    output = operation(left, right)
    upstream = rng.normal(size=output.shape).astype(np.float32)
    backward(output, upstream)
    expected = numerical_gradients(
        lambda a, b: np.sum(operation(a, b) * upstream), [left.data, right.data]
    )
    for tensor, gradient in zip((left, right), expected):
        assert tensor.grad.shape == tensor.shape
        np.testing.assert_allclose(tensor.grad, gradient, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "operation",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.pow,
        operator.matmul,
    ],
    ids=["add", "sub", "mul", "div", "pow", "matmul"],
)
@pytest.mark.parametrize("array_first", [False, True])
def test_numpy_operands_and_reflected_gradients(
    operation, array_first, numerical_gradients
):
    x = Tensor([[1.2, 1.5], [0.8, 2.0]], requires_grad=True)
    constant = np.array([[0.7, 1.1], [1.3, 0.9]], dtype=np.float32)

    def apply(value):
        return operation(constant, value) if array_first else operation(value, constant)

    output = apply(x)
    np.testing.assert_allclose(output.data, apply(x.data), rtol=1e-6)
    upstream = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    backward(output, upstream)
    (expected,) = numerical_gradients(
        lambda value: np.sum(apply(value) * upstream), [x.data]
    )
    assert x.grad.shape == x.shape
    np.testing.assert_allclose(x.grad, expected, rtol=2e-5, atol=2e-6)


def test_zero_to_constant_zero_power_has_zero_gradient():
    x = Tensor(0.0, requires_grad=True)
    # The function x**0 is constant, including the engine's forward at zero.
    with np.errstate(divide="raise", invalid="raise"):
        output = x**0
        backward(output)
    np.testing.assert_allclose(output.data, 1)
    np.testing.assert_allclose(x.grad, 0)


@pytest.mark.parametrize(
    "operation,expected",
    [
        (lambda x: 2 + x, [1.0, -2.0]),
        (lambda x: 2 - x, [-1.0, 2.0]),
        (lambda x: 2 * x, [2.0, -4.0]),
        (lambda x: 2 / x, [-0.5, 0.25]),
        (lambda x: 2**x, [4 * np.log(2), -32 * np.log(2)]),
    ],
)
def test_reflected_python_scalar_gradients(operation, expected):
    value = Tensor([2.0, 4.0], requires_grad=True)
    backward(operation(value), np.array([1.0, -2.0], dtype=np.float32))
    assert value.grad.shape == value.shape
    np.testing.assert_allclose(value.grad, expected, rtol=1e-6)


def test_negative_base_with_constant_integer_exponent_needs_no_log():
    x = Tensor([-2.0, -3.0], requires_grad=True)
    with np.errstate(invalid="raise"):
        backward(x**2, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(x.grad, [-4.0, -12.0])
