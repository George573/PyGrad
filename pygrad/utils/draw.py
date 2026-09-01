def _print_graph(tensor, include_gradients=False):
    """
    Print the computational graph starting from the given tensor.

    The graph shows:
    - Each tensor with its data and memory address
    - Operations that created tensors
    - Input tensors to each operation (recursively)
    """

    def tensor_label(node):
        label = repr(node)
        if include_gradients:
            label += f" | grad={getattr(node, 'grad', None)}"
        return label

    def visit(tensor, prefix="", is_last=True):
        # Print the tensor itself
        branch = "└── " if is_last else "├── "
        print(prefix + branch + tensor_label(tensor))

        # If this tensor was created by an operation, show the op and its inputs
        if tensor.op is not None:
            op = tensor.op
            new_prefix = prefix + ("    " if is_last else "│   ")

            # Print the operation
            print(new_prefix + "|")
            print(new_prefix + str(op) + f" ({hex(id(op))})")

            # Recursively visit input tensors
            inputs = tensor.inputs
            for i, inp in enumerate(inputs):
                visit(inp, new_prefix, i == len(inputs) - 1)

    # Start printing from the root tensor
    print(tensor_label(tensor))

    # If this tensor has an operation, print it and its inputs
    if tensor.op is not None:
        op = tensor.op
        print("|")
        print(str(op) + f" ({hex(id(op))})")

        # Visit all input tensors
        inputs = tensor.inputs
        for i, inp in enumerate(inputs):
            visit(inp, "", i == len(inputs) - 1)


def print_graph(tensor):
    """Print the computation graph rooted at ``tensor``."""
    _print_graph(tensor)


def print_graph_with_gradients(tensor):
    """Print the computation graph and each tensor's computed gradient."""
    _print_graph(tensor, include_gradients=True)
