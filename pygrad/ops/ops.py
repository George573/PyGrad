class Ops:
    def __init__(self, *inputs):
        self.inputs = inputs

    @staticmethod
    def create_tensor(data, op, input_tensors):
        from pygrad.tensor.tensor import Tensor

        output_tensor = Tensor(data, op=op, inputs=input_tensors)
        for tensor in input_tensors:
            tensor.outputs.append(output_tensor)
        return output_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.inputs))})"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def forward(self):
        raise NotImplementedError(f"Forward method not implemented {self!s}.")

    def backward(self, grad):
        raise NotImplementedError(f"Backward method not implemented {self!s}.")
