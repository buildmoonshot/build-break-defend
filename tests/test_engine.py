import math

from gpt_from_scratch.engine import Value


def test_forward_values():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c
    assert d.data == 4.0


def test_grads_match_hand_math():
    # the exact example from docs/pr1-computation-graph.svg
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c
    d.backward()
    assert d.grad == 1.0
    assert c.grad == 1.0    # bump c by 1 -> d rises by exactly 1
    assert a.grad == -3.0   # bump a by 1 -> d changes by b
    assert b.grad == 2.0    # bump b by 1 -> d changes by a


def test_reused_value_accumulates_grad():
    a = Value(3.0)
    d = a + a  # a is used twice, so its influence doubles
    d.backward()
    assert a.grad == 2.0


def test_tanh_grad():
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    assert math.isclose(y.data, math.tanh(0.5))
    assert math.isclose(x.grad, 1 - math.tanh(0.5) ** 2)


def _f(x):
    # an arbitrary lumpy function using several ops at once
    return (x * 2 + 1) * x + (x - 4) ** 2 + (x / 2).relu()


def test_grad_matches_numerical_estimate():
    # the lie detector: compare backward() against actually nudging the
    # input a hair and re-running the whole computation
    x = Value(1.7)
    y = _f(x)
    y.backward()

    h = 1e-6
    numeric = (_f(Value(1.7 + h)).data - _f(Value(1.7 - h)).data) / (2 * h)
    assert math.isclose(x.grad, numeric, rel_tol=1e-4)
