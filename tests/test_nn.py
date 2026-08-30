import random

from gpt_from_scratch.engine import Value
from gpt_from_scratch.nn import MLP, train_step


def test_forward_returns_value():
    random.seed(0)
    model = MLP(2, [3, 1])
    out = model([1.0, -1.0])
    assert isinstance(out, Value)
    assert -1.0 <= out.data <= 1.0  # tanh output range


def test_parameter_count():
    random.seed(0)
    # layer 1: 4 neurons x (2 weights + 1 bias) = 12
    # layer 2: 4 neurons x (4 weights + 1 bias) = 20
    # layer 3: 1 neuron  x (4 weights + 1 bias) = 5
    model = MLP(2, [4, 4, 1])
    assert len(model.parameters()) == 37


def test_zero_grad_clears_blame():
    random.seed(0)
    model = MLP(2, [2, 1])
    loss = (model([1.0, 0.0]) - 1.0) ** 2
    loss.backward()
    assert any(p.grad != 0.0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())


def test_training_reduces_loss():
    random.seed(42)
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [-1.0, 1.0, 1.0, -1.0]
    model = MLP(2, [4, 4, 1])

    first_loss = train_step(model, inputs, targets, learning_rate=0.1)
    for _ in range(99):
        last_loss = train_step(model, inputs, targets, learning_rate=0.1)
    assert last_loss < first_loss


def test_network_learns_xor():
    random.seed(42)
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [-1.0, 1.0, 1.0, -1.0]
    model = MLP(2, [4, 4, 1])

    for _ in range(300):
        train_step(model, inputs, targets, learning_rate=0.1)

    # every prediction must land on the correct side of zero
    for x, target in zip(inputs, targets):
        assert (model(x).data > 0) == (target > 0)
