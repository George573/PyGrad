import pygrad


def test_package_imports():
    assert pygrad.__version__ == "0.1.0"
    assert pygrad.tensor.Tensor is pygrad.Tensor


def test_tensor_addition():
    result = pygrad.tensor.Tensor(5) + pygrad.tensor.Tensor(10)

    assert result.data == 15


def test_tensor_subtraction():
    result = pygrad.tensor.Tensor(10) - pygrad.tensor.Tensor(3)

    assert result.data == 7
    assert isinstance(result.created, pygrad.ops.operations.Sub)


def test_tensor_multiplication():
    result = pygrad.tensor.Tensor(6) * pygrad.tensor.Tensor(4)

    assert result.data == 24
    assert isinstance(result.created, pygrad.ops.operations.Mul)


def test_tensor_matrix_multiplication():
    result = pygrad.tensor.Tensor([[1, 2], [3, 4]]) @ pygrad.tensor.Tensor(
        [[5, 6], [7, 8]]
    )

    assert result.data.tolist() == [[19, 22], [43, 50]]
    assert isinstance(result.created, pygrad.ops.operations.MatMul)


def test_tensor_division():
    result = pygrad.tensor.Tensor(10) / pygrad.tensor.Tensor(2)

    assert result.data == 5
    assert isinstance(result.created, pygrad.ops.operations.Div)


def test_tensor_negation():
    result = -pygrad.tensor.Tensor(5)

    assert result.data == -5
    assert isinstance(result.created, pygrad.ops.operations.Neg)


def test_tensor_power():
    result = pygrad.tensor.Tensor(2) ** pygrad.tensor.Tensor(3)

    assert result.data == 8
    assert isinstance(result.created, pygrad.ops.operations.Pow)


def test_tensor_sum():
    result = pygrad.tensor.Tensor([1, 2, 3]).sum()

    assert result.data == 6
    assert isinstance(result.created, pygrad.ops.operations.Sum)


def test_tensor_mean():
    result = pygrad.tensor.Tensor([[1, 2], [3, 4]]).mean(axis=0)

    assert result.data.tolist() == [2, 3]
    assert isinstance(result.created, pygrad.ops.operations.Mean)


def test_tensor_reshape():
    result = pygrad.tensor.Tensor([1, 2, 3, 4]).reshape((2, 2))

    assert result.data.tolist() == [[1, 2], [3, 4]]
    assert isinstance(result.created, pygrad.ops.operations.Reshape)


def test_tensor_flatten():
    result = pygrad.tensor.Tensor([[1, 2], [3, 4]]).flatten()

    assert result.data.tolist() == [1, 2, 3, 4]
    assert isinstance(result.created, pygrad.ops.operations.Flatten)


def test_tensor_transpose():
    result = pygrad.tensor.Tensor([[1, 2, 3], [4, 5, 6]]).transpose()

    assert result.data.tolist() == [[1, 4], [2, 5], [3, 6]]
    assert isinstance(result.created, pygrad.ops.operations.Transpose)
