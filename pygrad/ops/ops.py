class Ops:
    def __init__(self, *inputs):
        self.inputs = inputs

    @staticmethod
    def create_tensor(data, op, input_tensors):
        from pygrad.tensor.tensor import Tensor

        device = input_tensors[0].device
        if any(tensor.device != device for tensor in input_tensors[1:]):
            raise ValueError(
                "Cannot create an operation from tensors on different devices"
            )

        inputs = tuple(tensor for tensor in input_tensors if tensor.requires_grad)
        return Tensor(
            data,
            device=device,
            op=op,
            inputs=inputs or None,
            requires_grad=bool(inputs),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.inputs))})"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def forward(self):
        raise NotImplementedError(f"Forward method not implemented {self!s}.")

    def backward(self, grad):
        raise NotImplementedError(f"Backward method not implemented {self!s}.")
