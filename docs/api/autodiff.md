# Automatic differentiation API

## `backward`

```python
from pygrad.optimizers.backprop import backward
```

```python
backward(last_node, gradient=None)
```

Run reverse-mode automatic differentiation from an output tensor and store
gradient arrays in the `.grad` attributes of tracked tensors.

### Parameters

#### `last_node`

The output `Tensor` from which traversal begins.

When `gradient` is omitted, `last_node` must contain exactly one element. Its
gradient is initialized with an array of ones having the same shape and backend.

#### `gradient`

An optional upstream gradient. It is converted with the output's NumPy or CuPy
backend and must have exactly the same shape as `last_node`.

Use an explicit upstream gradient for multi-element outputs. A one-hot array—one
`1` and zeros elsewhere—selects the gradient of one output element.

### Return value

`None`. Gradients are written to tensors rather than returned.

### Exceptions

- `RuntimeError` if `gradient` is omitted for an output containing more than one
  element;
- `ValueError` if the supplied gradient shape differs from the output shape;
- backend conversion errors if the supplied gradient cannot be converted.

### Example

```python
from pygrad import Tensor
from pygrad.optimizers.backprop import backward

x = Tensor([2.0, 3.0], requires_grad=True)
loss = (x * x).sum()

backward(loss)
print(x.grad) # [4. 6.]
```

### Gradient lifetime

`backward()` does not clear gradients before traversal. The output gradient is
assigned from the new seed, while gradients already present on tracked inputs
are accumulated. Clear parameter gradients before a new optimization iteration;
`SGD.zero_grad()` does this for parameters managed by that optimizer.

`pending_grads()` is an internal graph-scheduling helper and is not part of the
documented user API.
