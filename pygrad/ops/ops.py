class Ops:
    def __init__(self, *inputs):
        self.inputs = inputs

    @staticmethod
    def create_tensor(data, created):
        from pygrad.tensor.tensor import Tensor

        return Tensor(data, created=created)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.inputs))})"
    
    def forward(self):
        raise NotImplementedError("Forward method not implemented.")