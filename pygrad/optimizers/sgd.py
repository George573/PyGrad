class SGD:
    def __init__(self, trainble_params, epcilon=1e-3):
        self.traible_params = self.traible_params
        self.epcilon = epcilon
        
    def step(self, grad_table: dict):
        for param in self.traible_params:
            grad = grad_table.get(param)
            if not grad:
                raise ValueError(
                    f"Parametr {param} doesn't have a computed gradient"
                )
            param.data -= self.epcilon * grad