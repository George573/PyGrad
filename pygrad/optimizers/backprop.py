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
    last_node.grad = gradient

    wait_queue = deque()
    wait_queue.append(last_node)
    pending = pending_grads(last_node)

    while wait_queue:
        n = wait_queue.popleft()

        if n.op is None:
            continue
        input_grad = n.op.backward(n.grad)
        for p, p_grad in zip(n.op.inputs, input_grad):
            if not p.requires_grad:
                continue
            if p.grad is not None:
                p.grad = p.grad + p_grad
            else:
                p.grad = p_grad
            pending[p] -= 1
            if pending[p] == 0:
                wait_queue.append(p)
