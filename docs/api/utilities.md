# Graph utilities

PyGrad provides text-based computation-graph inspection functions. They print
to standard output and return `None`.

## `print_graph`

```python
from pygrad.utils.draw import print_graph
```

```python
print_graph(tensor)
```

Print the tracked graph rooted at `tensor`. The output shows:

- each tracked tensor's representation and memory address;
- the operation that created each result;
- the tracked inputs to each operation.

```python
from pygrad import Tensor
from pygrad.utils.draw import print_graph

x = Tensor(2.0, requires_grad=True)
y = x * 3

print_graph(y)
```

Only inputs retained in the backward graph are displayed. Constants that do not
require gradients are omitted from a result's tracked `inputs`.

## `print_graph_with_gradients`

```python
from pygrad.utils.draw import print_graph_with_gradients
```

```python
print_graph_with_gradients(tensor)
```

Print the same graph with the current `.grad` value beside every tensor.

```python
from pygrad import Tensor
from pygrad.optimizers.backprop import backward
from pygrad.utils.draw import print_graph_with_gradients

x = Tensor(2.0, requires_grad=True)
y = x * 3
backward(y)

print_graph_with_gradients(y)
```

Before a backward pass, tracked gradients appear as `None`. Object addresses and
formatting details are diagnostic output and may vary between runs.
