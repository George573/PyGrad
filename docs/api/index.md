# API reference

This reference describes PyGrad's current public interfaces and exact call
signatures.

- [`Tensor`](tensor.md) — data, gradient tracking, operators, transformations,
  reductions, and element-wise methods
- [`backward`](autodiff.md) — reverse-mode differentiation
- [`SGD`](optimizers.md) — parameter updates, momentum, and gradient clearing
- [Graph utilities](utilities.md) — print computation graphs and gradients

The classes in `pygrad.ops` implement PyGrad internally. Normal user code should
call operations through `Tensor` operators and methods rather than instantiate
operation classes directly.
