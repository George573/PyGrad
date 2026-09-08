class SGD:
    def __init__(self, trainable_params, epsilon=1e-3, momentum=0.0):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in the range [0, 1]")

        self.trainable_params = list(trainable_params)
        self.epsilon = epsilon
        self.momentum = momentum
        self.velocity_table = {}

    def step(self):
        for param in self.trainable_params:
            grad = getattr(param, "grad", None)
            if grad is None:
                raise ValueError(f"Parameter {param} doesn't have a computed gradient")

            prev_v = self.velocity_table.get(param, 0)

            v = self.momentum * prev_v - self.epsilon * grad

            self.velocity_table[param] = v
            param.data += v

    def zero_grad(self):
        for param in self.trainable_params:
            param.grad = None
