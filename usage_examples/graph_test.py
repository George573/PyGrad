import pygrad

a = pygrad.tensor.Tensor([5, 2, 3])
b = pygrad.tensor.Tensor([10, 20, 30])

c = a + b

g = a + b + c

b = g * a - pygrad.tensor.Tensor(5)

b = g.reshape((-1, 1)) @ a.reshape((1, -1))



# print(c.created)
# print(g.created)

# print(a, b, c, g)
# print(g.created.inputs[0].created)

# print(g.created.inputs[0].data.data)

pygrad.utils.draw.print_graph(b)

