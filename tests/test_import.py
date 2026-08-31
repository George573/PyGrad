import pygrad


def test_package_imports():
    assert pygrad.__version__ == "0.1.0"
    assert pygrad.tensor.Tensor is pygrad.Tensor
