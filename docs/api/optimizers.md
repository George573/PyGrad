# Optimizer API

## `SGD`

```python
from pygrad.optimizers.sgd import SGD
```

```python
SGD(trainable_params, epsilon=1e-3, momentum=0.0)
```

Update parameters with stochastic gradient descent and optional momentum.

### Parameters

#### `trainable_params`

An iterable of tensors to update. Each parameter should have
`requires_grad=True` and a computed `.grad` before `step()` is called.

#### `epsilon`

The learning rate. Defaults to `1e-3` and must be greater than zero. A
non-positive value raises `ValueError`.

#### `momentum`

The momentum coefficient. Defaults to `0.0` and must satisfy
`0 <= momentum < 1`. Values outside that range raise `ValueError`.

### Update rule

For each parameter, `step()` calculates:

```text
velocity = momentum * previous_velocity - epsilon * gradient
parameter = parameter + velocity
```

Velocity starts at zero and is retained by the optimizer between steps.

### `step()`

Update every managed parameter using its current `.grad`. Returns `None`.

If a parameter has no computed gradient, `step()` raises `ValueError`. The
method mutates each parameter's underlying `.data` array in place.

### `zero_grad()`

Set `.grad` to `None` on every managed parameter. Returns `None`.

Call it before computing gradients for the next optimization iteration so old
parameter gradients are not accumulated.

### Example

```python
from pygrad import Tensor
from pygrad.optimizers.backprop import backward
from pygrad.optimizers.sgd import SGD

weight = Tensor([2.0], requires_grad=True)
optimizer = SGD([weight], epsilon=0.1, momentum=0.0)

optimizer.zero_grad()
error = weight - 5.0
loss = error * error
backward(loss)
optimizer.step()

print(weight.data) # [2.6]
```
