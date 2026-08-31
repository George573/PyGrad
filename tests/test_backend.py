import numpy as np
import pytest

from pygrad.backend.backend import as_array, get_array_module, is_array


def test_cpu_backend_converts_values_to_numpy_array():
    result = as_array([1.0, 2.0], device="cpu")

    np.testing.assert_allclose(result, [1.0, 2.0])
    assert isinstance(result, np.ndarray)
    assert is_array(result)
    assert get_array_module(result) is np


def test_backend_rejects_unknown_device():
    with pytest.raises(ValueError, match="Unknown device: quantum"):
        as_array([1.0], device="quantum")


def test_array_module_rejects_non_array():
    with pytest.raises(TypeError, match="Unknown array type"):
        get_array_module([1.0, 2.0])
