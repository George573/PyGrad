from pygrad.ops.ops import Ops

class Add(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data + b.data
    
    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, created=self)
    
class Sub(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data - b.data
    
    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, created=self)

class Mul(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data * b.data
    
    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, created=self)

class MatMul(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data @ b.data
    
    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, created=self)


class Div(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data / b.data

    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, created=self)


class Neg(Ops):
    def forward(self):
        (a,) = self.inputs
        return -a.data

    def __call__(self, a):
        self.inputs = (a,)
        result = self.forward()
        return self.create_tensor(result, created=self)


class Pow(Ops):
    def forward(self):
        a, b = self.inputs
        return a.data ** b.data

    def __call__(self, a, b):
        self.inputs = (a, b)
        result = self.forward()
        return self.create_tensor(result, created=self)
    
class Reshape(Ops):
    def forward(self):
        (a, ) = self.inputs
        return a.data.reshape(self.shape)

    def __call__(self, a, b):
        self.inputs = (a,)
        self.shape = b
        result = self.forward()
        return self.create_tensor(result, created=self)
    
class Flatten(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.flatten()

    def __call__(self, a):
        self.inputs = (a,)
        result = self.forward()
        return self.create_tensor(result, created=self)
    
class Transpose(Ops):
    def forward(self):
        (a, ) = self.inputs
        return a.data.transpose(self.axes)

    def __call__(self, a, axes=None):
        self.inputs = (a,)
        self.axes = axes
        result = self.forward()
        return self.create_tensor(result, created=self)


class Sum(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.sum(axis=self.axis, keepdims=self.keepdims)

    def __call__(self, a, axis=None, keepdims=False):
        self.inputs = (a,)
        self.axis = axis
        self.keepdims = keepdims
        result = self.forward()
        return self.create_tensor(result, created=self)


class Mean(Ops):
    def forward(self):
        (a,) = self.inputs
        return a.data.mean(axis=self.axis, keepdims=self.keepdims)

    def __call__(self, a, axis=None, keepdims=False):
        self.inputs = (a,)
        self.axis = axis
        self.keepdims = keepdims
        result = self.forward()
        return self.create_tensor(result, created=self)