class SGD:
    def __init__(self, trainble_params, epcilon=1e-3):
        self.traible_params = trainble_params
        self.epcilon = epcilon

    def step(self):
        for param in self.traible_params:
            grad = getattr(param, "grad", None)
            if grad is None:
                raise ValueError(f"Parametr {param} doesn't have a computed gradient")
            param.data -= self.epcilon * grad
            
    def zero_grad(self):
        for param in self.traible_params:
            param.grad = None
