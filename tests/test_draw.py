import numpy as np

from pygrad import Tensor
from pygrad.optimizers.backprop import backward
from pygrad.utils.draw import print_graph_with_gradients


def test_print_graph_with_gradients(capsys):
    value = Tensor(2.0, requires_grad=True)
    result = value * 3
    backward(result)

    print_graph_with_gradients(result)

    output = capsys.readouterr().out
    assert "grad=1.0" in output
    assert "grad=3.0" in output
    assert "Mul" in output


def test_print_graph_with_gradients_shows_uncomputed_gradient(capsys):
    result = Tensor(np.array([1.0, 2.0]), requires_grad=True).sum()

    print_graph_with_gradients(result)

    assert "grad=None" in capsys.readouterr().out
