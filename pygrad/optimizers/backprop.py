from collections import defaultdict, deque


def pending_grads(last_node):
    pending = defaultdict(int)
    wait_queue = deque()
    wait_queue.append(last_node)

    visited = set()

    while wait_queue:
        n = wait_queue.popleft()

        if n in visited:
            continue

        visited.add(n)

        for parent in n.inputs:
            pending[parent] += 1
            wait_queue.append(parent)

    return pending


def backward(last_node, gradient=None):
    from pygrad.backend.backend import get_array_module

    xp = get_array_module(last_node.data)

    if gradient is None:
        if last_node.size != 1:
            raise RuntimeError(
                "gradient must be provided for outputs with multiple elements"
            )

        gradient = xp.ones_like(last_node.data)

    else:
        gradient = xp.asanyarray(gradient)

        if gradient.shape != last_node.shape:
            raise ValueError(
                f"gradient shape {gradient.shape} does not match "
                f"output shape {last_node.shape}"
            )

    wait_queue = deque()
    wait_queue.append(last_node)
    pending = pending_grads(last_node)

    grad_table = {last_node: gradient}

    while wait_queue:
        n = wait_queue.popleft()

        if n.op is None:
            continue
        input_grad = n.op.backward(grad_table[n])
        if len(n.op.inputs) != len(input_grad):
            raise RuntimeError(
                f"Node: {n} has {len(n.op.inputs)} inputs"
                f" but {len(input_grad)} input gradients"
                f" {n.op!s} must have a bug"
            )
        for p, p_grad in zip(n.op.inputs, input_grad):
            if not p.requires_grad:
                continue
            if p.shape != p_grad.shape:
                raise RuntimeError(
                    f"Gradient shape mismatch: "
                    f"Node {p!s} has shape: {p.shape}"
                    f", when it's gradient: {p_grad.shape}"
                )
            if p in grad_table:
                grad_table[p] = grad_table[p] + p_grad
            else:
                grad_table[p] = p_grad
            pending[p] -= 1
            if pending[p] == 0:
                wait_queue.append(p)

    for node, node_grad in grad_table.items():
        node.grad = node_grad if node.grad is None else node.grad + node_grad
