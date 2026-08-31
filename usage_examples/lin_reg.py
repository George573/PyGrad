import pygrad
from pygrad.tensor import Tensor

y = pygrad.tensor.Tensor([5, 2, 3]).reshape((1, -1)) @ pygrad.tensor.Tensor(
    [10, 20, 30]
) + Tensor(5)

pygrad.utils.draw.print_graph(y)
