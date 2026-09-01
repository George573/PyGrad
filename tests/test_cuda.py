import numpy as np
import pytest

from pygrad import Tensor
from pygrad.optimizers.backprop import backward

cp = pytest.importorskip("cupy", reason="CuPy is required for CUDA tests")
pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def cuda_device():
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")

    if device_count == 0:
        pytest.skip("No CUDA-capable device is available")

    with cp.cuda.Device(0):
        yield


def assert_gpu_tensor(tensor, expected, **kwargs):
    assert tensor.device == "cuda"
    assert isinstance(tensor.data, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(tensor.data), expected, **kwargs)


def test_cuda_tensor_creation():
    tensor = Tensor([1.0, 2.0, 3.0], device="cuda")

    assert_gpu_tensor(tensor, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda left, right: left + right, [5.0, 7.0, 9.0]),
        (lambda left, right: left - right, [-3.0, -3.0, -3.0]),
        (lambda left, right: left * right, [4.0, 10.0, 18.0]),
        (lambda left, right: left / right, [0.25, 0.4, 0.5]),
        (lambda left, right: left**right, [1.0, 32.0, 729.0]),
    ],
)
def test_cuda_tensor_arithmetic(operation, expected):
    left = Tensor([1.0, 2.0, 3.0], device="cuda")
    right = Tensor([4.0, 5.0, 6.0], device="cuda")

    assert_gpu_tensor(operation(left, right), expected)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda value: value + 2, [3.0, 4.0, 5.0]),
        (lambda value: 2 + value, [3.0, 4.0, 5.0]),
        (lambda value: value - 2, [-1.0, 0.0, 1.0]),
        (lambda value: 2 - value, [1.0, 0.0, -1.0]),
        (lambda value: value * 2, [2.0, 4.0, 6.0]),
        (lambda value: 2 * value, [2.0, 4.0, 6.0]),
        (lambda value: value / 2, [0.5, 1.0, 1.5]),
        (lambda value: 2 / value, [2.0, 1.0, 2.0 / 3.0]),
        (lambda value: value**2, [1.0, 4.0, 9.0]),
        (lambda value: 2**value, [2.0, 4.0, 8.0]),
    ],
)
def test_cuda_scalar_arithmetic_preserves_device(operation, expected):
    value = Tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)

    assert_gpu_tensor(operation(value), expected)


@pytest.mark.parametrize(
    ("method", "values", "expected"),
    [
        ("exp", [-1.0, 0.0, 1.0], np.exp([-1.0, 0.0, 1.0])),
        ("log", [0.5, 1.0, 2.0], np.log([0.5, 1.0, 2.0])),
        ("sqrt", [0.25, 1.0, 4.0], [0.5, 1.0, 2.0]),
        ("abs", [-2.0, 0.0, 3.0], [2.0, 0.0, 3.0]),
        ("tanh", [-1.0, 0.0, 1.0], np.tanh([-1.0, 0.0, 1.0])),
        ("sigmoid", [-1000.0, 0.0, 1000.0], [0.0, 0.5, 1.0]),
        ("relu", [-2.0, 0.0, 3.0], [0.0, 0.0, 3.0]),
    ],
)
def test_cuda_elementwise_operations(method, values, expected):
    value = Tensor(values, device="cuda")

    assert_gpu_tensor(getattr(value, method)(), expected, rtol=1e-6, atol=1e-7)


def test_cuda_reduction_and_shape_operations():
    value = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")

    assert_gpu_tensor(value.sum(axis=0), [4.0, 6.0])
    assert_gpu_tensor(value.mean(), 2.5)
    assert_gpu_tensor(value.reshape(4).transpose().flatten(), [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 2.0], [3.0, 4.0], 11.0),
        ([1.0, 2.0], [[3.0, 4.0], [5.0, 6.0]], [13.0, 16.0]),
        ([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0], [17.0, 39.0]),
        (
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[19.0, 22.0], [43.0, 50.0]],
        ),
    ],
)
def test_cuda_matmul_forward(left, right, expected):
    result = Tensor(left, device="cuda") @ Tensor(right, device="cuda")

    assert_gpu_tensor(result, expected)


def test_cuda_composed_backward_and_gradient_device():
    value = Tensor([1.0, 2.0, 3.0], device="cuda")
    sigmoid = value.sigmoid()
    loss = (sigmoid * value + 2).mean()

    backward(loss)
    sigmoid_cpu = 1 / (1 + np.exp(-np.array([1.0, 2.0, 3.0])))
    expected = (
        sigmoid_cpu + np.array([1.0, 2.0, 3.0]) * sigmoid_cpu * (1 - sigmoid_cpu)
    ) / 3

    assert isinstance(value.grad, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(value.grad), expected, rtol=1e-6, atol=1e-7)


def test_cuda_matrix_vector_backward():
    left = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", requires_grad=True)
    right = Tensor([5.0, 6.0], device="cuda", requires_grad=True)

    backward((left @ right).sum())

    assert isinstance(left.grad, cp.ndarray)
    assert isinstance(right.grad, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(left.grad), [[5.0, 6.0], [5.0, 6.0]])
    np.testing.assert_allclose(cp.asnumpy(right.grad), [4.0, 6.0])
