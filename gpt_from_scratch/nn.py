"""Tiny neural network built on the PR-1 autograd engine (PR 2).

A Neuron is a handful of Value knobs: weights and a bias. It multiplies
each input by a weight, sums them, and squashes through tanh. A Layer is
a row of neurons; an MLP is a stack of layers. Because every knob is a
Value, calling .backward() on a loss automatically fills in every knob's
blame — training is then just "nudge each knob against its grad".

See docs/pr2-training-loop.svg for the picture version.
"""

import random

from gpt_from_scratch.engine import Value


class Module:
    """Shared plumbing: list your knobs, and reset their blame between steps."""

    def parameters(self):
        return []

    def zero_grad(self):
        # grads accumulate with += during backward(), so they must be
        # cleared before every training step or blame from old steps leaks in
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        total = self.b
        for wi, xi in zip(self.w, x):
            total = total + wi * xi
        return total.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Multi-layer perceptron: MLP(2, [4, 4, 1]) = 2 inputs -> 4 -> 4 -> 1 output."""

    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def train_step(model, inputs, targets, learning_rate):
    """One full turn of the loop: predict, measure, blame, nudge. Returns the loss."""
    # predict (forward pass)
    predictions = [model(x) for x in inputs]
    # measure: one wrongness score for the whole batch
    loss = Value(0.0)
    for pred, target in zip(predictions, targets):
        loss = loss + (pred - target) ** 2
    # blame (backward pass)
    model.zero_grad()
    loss.backward()
    # nudge every knob a tiny step AGAINST its blame
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    return loss.data
