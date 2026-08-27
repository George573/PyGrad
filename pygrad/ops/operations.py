from pygrad.ops.ops import Ops

class Add(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data + b.data
    
    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, created=self)