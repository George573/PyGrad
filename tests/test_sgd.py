import numpy as np
import pytest

from pygrad import Tensor
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
