import numpy as np
import pytest

from pygrad import Tensor
from pygrad.optimizers.backprop import backward
from pygrad.optimizers.sgd import SGD


def test_sgd_accepts_correctly_spelled_keyword_arguments():
    parameter = Tensor(1.0, requires_grad=True)

    optimizer = SGD([parameter], epsilon=0.1, momentum=0.5)

    assert optimizer.trainable_params == [parameter]
    assert optimizer.epsilon == 0.1
    assert optimizer.momentum == 0.5


@pytest.mark.parametrize("epsilon", [0.0, -0.1])
def test_sgd_rejects_non_positive_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon must be positive"):
        SGD([], epsilon=epsilon)


@pytest.mark.parametrize("momentum", [-0.1, 1.0, 2.0])
def test_sgd_rejects_invalid_momentum(momentum):
    with pytest.raises(ValueError, match=r"momentum must be in the range \[0, 1\]"):
        SGD([], momentum=momentum)


def test_sgd_step_updates_parameter_with_momentum():
    parameter = Tensor(2.0, requires_grad=True)
    optimizer = SGD([parameter], epsilon=0.1, momentum=0.5)

    parameter.grad = np.array(3.0, dtype=np.float32)
    optimizer.step()
    np.testing.assert_allclose(parameter.data, 1.7)

    parameter.grad = np.array(2.0, dtype=np.float32)
    optimizer.step()
    np.testing.assert_allclose(parameter.data, 1.35)


def test_sgd_step_requires_a_gradient():
    parameter = Tensor(2.0, requires_grad=True)

    with pytest.raises(ValueError, match="doesn't have a computed gradient"):
        SGD([parameter]).step()


def test_sgd_zero_grad_clears_parameter_gradients():
    parameter = Tensor(2.0, requires_grad=True)
    parameter.grad = np.array(1.0, dtype=np.float32)

    SGD([parameter]).zero_grad()

    assert parameter.grad is None


@pytest.mark.parametrize("clear_before_step", [False, True])
def test_sgd_preserves_generator_parameters_across_calls(clear_before_step):
    parameter = Tensor(2.0, requires_grad=True)
    optimizer = SGD((p for p in [parameter]), epsilon=0.1)
    if clear_before_step:
        optimizer.zero_grad()
    for expected in [1.9, 1.8]:
        parameter.grad = np.array(1.0, dtype=np.float32)
        optimizer.step()
        np.testing.assert_allclose(parameter.data, expected)


def test_sgd_keeps_momentum_independent_for_multiple_parameters():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor([-1.0, 4.0], requires_grad=True)
    optimizer = SGD([a, b], epsilon=0.1, momentum=0.5)
    for grad_a, grad_b, expected_a, expected_b in [
        (3.0, [2.0, -1.0], 1.7, [-1.2, 4.1]),
        (2.0, [-1.0, 2.0], 1.35, [-1.2, 3.95]),
    ]:
        a.grad = np.array(grad_a, dtype=np.float32)
        b.grad = np.array(grad_b, dtype=np.float32)
        optimizer.step()
        np.testing.assert_allclose(a.data, expected_a, rtol=1e-6)
        np.testing.assert_allclose(b.data, expected_b, rtol=1e-6)
    optimizer.zero_grad()
    assert a.grad is None
    assert b.grad is None


def test_linear_regression_training_matches_numpy_updates():
    features = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
    targets = 2 * features + 1
    weights = Tensor([[0.0]], requires_grad=True)
    bias = Tensor([0.0], requires_grad=True)
    optimizer = SGD([weights, bias], epsilon=0.05)
    expected_weights = np.zeros((1, 1), dtype=np.float64)
    expected_bias = np.zeros(1, dtype=np.float64)
    initial_loss = float(np.mean(targets**2))
    for _ in range(10):
        optimizer.zero_grad()
        prediction = Tensor(features) @ weights + bias
        error = prediction - Tensor(targets)
        loss = (error * error).mean()
        backward(loss)

        residual = features @ expected_weights + expected_bias - targets
        grad_weights = 2 * features.T @ residual / features.shape[0]
        grad_bias = 2 * residual.mean(axis=0)
        np.testing.assert_allclose(weights.grad, grad_weights, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bias.grad, grad_bias, rtol=1e-5, atol=1e-6)
        expected_weights -= 0.05 * grad_weights
        expected_bias -= 0.05 * grad_bias
        optimizer.step()
        np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bias.data, expected_bias, rtol=1e-5, atol=1e-6)
    final_loss = np.mean((features @ weights.data + bias.data - targets) ** 2)
    assert final_loss < initial_loss * 0.1
