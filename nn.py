import random
from micrograd import Value
from numpy import multiply

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    __slots__ = ['w', 'b', 'non_linear']

    def __init__(self, n_in: int, non_linear: bool = True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Value(0)
        self.non_linear = non_linear

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return activation.relu() if self.non_linear else activation

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.non_linear else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    __slots__ = ['neurons']

    def __init__(self, n_in:int, n_out:int, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    __slots__ = ['layers']

    def __init__(self, n_in: int, n_out):
        sz = [n_in] + n_out
        last_node = len(n_out) - 1
        self.layers = [Layer(sz[i], sz[i+1], non_linear=(i != last_node)) for i in range(len(n_out))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

