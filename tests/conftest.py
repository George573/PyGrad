import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-cuda",
        action="store_true",
        default=False,
        help="Skip tests marked cuda, even when a GPU is available.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-cuda"):
        skip_cuda = pytest.mark.skip(reason="CUDA tests disabled by --skip-cuda")
        for item in items:
            if item.get_closest_marker("cuda") is not None:
                item.add_marker(skip_cuda)


@pytest.fixture
def numerical_gradients():
    """Differentiate an independent scalar NumPy reference in float64.

    Pass actual tensor data so the reference starts at the engine's rounded
    input values. Perturbations never pass through Tensor's float32 conversion.
    """

    def compute(objective, inputs):
        values = [np.array(value, dtype=np.float64, copy=True) for value in inputs]
        gradients = []
        for value in values:
            expected = np.empty_like(value)
            for index in np.ndindex(value.shape):
                original = value[index]
                step = 1e-5 * max(1.0, abs(original))
                value[index] = original + step
                plus = objective(*values)
                value[index] = original - step
                minus = objective(*values)
                value[index] = original
                expected[index] = (plus - minus) / (2 * step)
            gradients.append(expected)
        return gradients

    return compute
