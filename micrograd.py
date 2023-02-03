import math
from functools import partial


def add_backward(self, other, out):
    self.grad += out.grad
    other.grad += out.grad


def multiply_backward(self, other, out):
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad


def power_backward(self, other, out):
    self.grad += (other * self.data**(other-1)) * out.grad


def relu_backward(self, out):
    self.grad += (out.data > 0) * out.grad


def tanh_backward(self, t, out):
    self.grad += (1 - t ** 2) * out.grad


def exp_backward(self, out):
    self.grad += out.data * out.grad


def build_topo(v, visited, topo):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child, visited, topo)
        topo.append(v)


class Value:
    """ Stores a single scalar value and its gradient """
    __slots__ = ['data', 'grad', '_backward', '_prev', '_op']

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0

        # internal variables used for autograd graph construction
        self._backward = None  # lambda: None
        self._prev = _children #set(_children)
        self._op = _op  # the op that produced this code, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        out._backward = partial(add_backward, self, other, out)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        out._backward = partial(multiply_backward, self, other, out)

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        out._backward = partial(power_backward, self, other, out)

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        out._backward = partial(relu_backward, self, out)
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        out._backward = partial(tanh_backward, self, t, out)
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        out._backward = partial(exp_backward, self, out)
        return out

    def backward(self):

        # topological order of the children in the graph
        topo = []
        visited = set()

        build_topo(self, visited, topo)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            if v._backward:
                v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return other * self**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
