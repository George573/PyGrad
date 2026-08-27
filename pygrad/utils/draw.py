def print_graph(tensor):
    def visit(tensor, prefix="", is_last=True):
        branch = "└── " if is_last else "├── "
        print(prefix + branch + repr(tensor))

        if tensor.created is None:
            return

        op = tensor.created
        new_prefix = prefix + ("    " if is_last else "│   ")
        print(new_prefix + "|")
        print(new_prefix + str(op) + f' ({hex(id(op))})')

        for i, inp in enumerate(op.inputs):
            visit(inp, new_prefix, i == len(op.inputs) - 1)

    print(repr(tensor))
    if tensor.created is None:
        return

    op = tensor.created
    print("|")
    print(str(op) + f' ({hex(id(op))})')
    for i, inp in enumerate(op.inputs):
        visit(inp, "", i == len(op.inputs) - 1)