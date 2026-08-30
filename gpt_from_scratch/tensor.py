"""A tensor autograd engine (PR 4) — PR 1's idea, upgraded to grids.

Same soul as engine.py: a Tensor is a grid of numbers that remembers how
it was made, and backward() walks that history filling in blame. The
difference is scale — one Tensor op moves thousands of numbers at once,
with numpy doing the raw arithmetic. numpy is only a fast calculator:
every learning rule here (what blame each operation passes back) is
hand-written, same as PR 1, and verified against numerical estimates.
"""

import numpy as np


class Tensor:
    def __init__(self, data, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = _parents
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _unbroadcast(grad, shape):
        """When a small tensor was stretched (broadcast) to fit a big one,
        its blame from every copy must be summed back into the original."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for axis, size in enumerate(shape):
            if size == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # same swap rule as PR 1's multiply, in grid form
            ga = out.grad @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += Tensor._unbroadcast(ga, self.data.shape)
            other.grad += Tensor._unbroadcast(gb, other.data.shape)
        out._backward = _backward
        return out

    def transpose_last(self):
        out = Tensor(np.swapaxes(self.data, -1, -2), (self,), "T")

        def _backward():
            self.grad += np.swapaxes(out.grad, -1, -2)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def softmax(self):
        """Turn each last-axis row into probabilities that sum to 1."""
        shifted = self.data - self.data.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        p = e / e.sum(axis=-1, keepdims=True)
        out = Tensor(p, (self,), "softmax")

        def _backward():
            dot = (out.grad * p).sum(axis=-1, keepdims=True)
            self.grad += (out.grad - dot) * p
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        ordered, visited = [], set()

        def visit(t):
            if id(t) not in visited:
                visited.add(id(t))
                for parent in t._parents:
                    visit(parent)
                ordered.append(t)
        visit(self)

        self.grad = np.ones_like(self.data)
        for t in reversed(ordered):
            t._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"


def embedding(table, idx):
    """Look up rows of `table` by integer index — how text becomes vectors."""
    idx = np.asarray(idx)
    out = Tensor(table.data[idx], (table,), "embed")

    def _backward():
        # a row used many times collects blame from every use (PR 1's += rule)
        np.add.at(table.grad, idx, out.grad)
    out._backward = _backward
    return out


def cross_entropy(logits, targets):
    """Average surprise (negative log likelihood) — PR 3's loss, differentiable.

    logits: (..., V) scores per character; targets: matching integer array.
    """
    t = np.asarray(targets).reshape(-1)
    flat = logits.data.reshape(-1, logits.data.shape[-1])
    shifted = flat - flat.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    p = e / e.sum(axis=1, keepdims=True)
    n = flat.shape[0]
    out = Tensor(-np.log(p[np.arange(n), t] + 1e-12).mean(), (logits,), "ce")

    def _backward():
        g = p.copy()
        g[np.arange(n), t] -= 1.0  # the famous "probabilities minus truth"
        logits.grad += (g / n).reshape(logits.data.shape) * out.grad
    out._backward = _backward
    return out
