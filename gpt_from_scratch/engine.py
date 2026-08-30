"""A tiny automatic-differentiation engine (PR 1 of build-break-defend).

The one idea: a Value is a number that remembers how it was made.
Do math with Values and they quietly build a graph of every operation.
Call .backward() on the final result and it walks that graph in reverse,
filling in each Value's .grad — "if this number nudged up a tiny bit,
how much would the final result change?"

Training a neural network is nothing more than: compute grads, nudge
every knob a tiny step against its grad, repeat.

See docs/pr1-computation-graph.svg for the picture version.
"""

import math


class Value:
    """One number in the graph, plus the breadcrumbs of its history."""

    def __init__(self, data, _parents=(), _op=""):
        self.data = data                    # the actual number
        self.grad = 0.0                     # d(result)/d(this), filled by backward()
        self._parents = set(_parents)       # the Values this one was made from
        self._op = _op                      # the operation that made it
        self._backward = lambda: None       # how to pass grad back to parents

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # addition passes the grad through to both parents unchanged
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # nudging one factor moves the product by the OTHER factor
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only plain-number exponents"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            # relu lets grad through only where the input was positive
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Fill in .grad for every Value that fed into this one."""
        # 1. Order the graph so every value comes after its parents.
        ordered = []
        visited = set()

        def visit(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    visit(parent)
                ordered.append(v)
        visit(self)

        # 2. The result moves 1-for-1 with itself; walk history in reverse.
        self.grad = 1.0
        for v in reversed(ordered):
            v._backward()

    # conveniences so Values mix freely with plain numbers
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
