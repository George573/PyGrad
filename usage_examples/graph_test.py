import pygrad

a = pygrad.tensor.Tensor(5)
b = pygrad.tensor.Tensor(10)

c = a + b

g = a + b + c

print(c.created)
print(g.created)

print(a, b, c, g)
print(g.created.inputs[0].created)

print(g.created.inputs[0].data.shape)

