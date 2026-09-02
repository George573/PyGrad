class SGD:
    def __init__(self, trainble_params, epcilon=1e-3, momentum=0.0):
        if epcilon <= 0:
            raise ValueError('epcilon must be positive')
        
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in the range [0, 1]")
        
        self.traible_params = trainble_params
        self.epcilon = epcilon
        self.momentum = momentum
        self.velocity_table = {}

    def step(self):
        for param in self.traible_params:
            grad = getattr(param, "grad", None)
            if grad is None:
                raise ValueError(f"Parametr {param} doesn't have a computed gradient")
            
            prev_v = self.velocity_table.get(param, 0)
            
            v = (
                self.momentum * prev_v 
                - self.epcilon * grad
            )
            
            self.velocity_table[param] = v
            param.data += v
            
    def zero_grad(self):
        for param in self.traible_params:
            param.grad = None
