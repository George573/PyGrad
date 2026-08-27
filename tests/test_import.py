import pygrad


def test_package_imports():
    assert pygrad.__version__ == "0.1.0"
    assert pygrad.tensor.Tensor is pygrad.Tensor


def test_tensor_addition():
    result = pygrad.tensor.Tensor(5) + pygrad.tensor.Tensor(10)

    assert result.data == 15
