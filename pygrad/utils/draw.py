def print_graph(tensor):
    """
    Print the computational graph starting from the given tensor.

    The graph shows:
    - Each tensor with its data and memory address
    - Operations that created tensors
    - Input tensors to each operation (recursively)
    """

    def visit(tensor, prefix="", is_last=True):
        # Print the tensor itself
        branch = "└── " if is_last else "├── "
        print(prefix + branch + repr(tensor))

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
    print(repr(tensor))

    # If this tensor has an operation, print it and its inputs
    if tensor.op is not None:
        op = tensor.op
        print("|")
        print(str(op) + f" ({hex(id(op))})")

        # Visit all input tensors
        inputs = tensor.inputs
        for i, inp in enumerate(inputs):
            visit(inp, "", i == len(inputs) - 1)
