import numpy as np

from gpt_from_scratch.gpt import Adam, TinyGPT, sample_word
from gpt_from_scratch.tensor import Tensor, cross_entropy, embedding


def numeric_check(build_loss, tensor, spots=3, h=1e-6, tol=1e-4):
    """The lie detector, grid edition: nudge entries for real, compare slopes."""
    loss = build_loss()
    loss.backward()
    analytic = tensor.grad.copy()
    rng = np.random.default_rng(0)
    flat_idx = rng.choice(tensor.data.size, size=min(spots, tensor.data.size), replace=False)
    for i in flat_idx:
        idx = np.unravel_index(i, tensor.data.shape)
        old = tensor.data[idx]
        tensor.data[idx] = old + h
        up = build_loss().data
        tensor.data[idx] = old - h
        down = build_loss().data
        tensor.data[idx] = old
        numeric = (up - down) / (2 * h)
        assert abs(analytic[idx] - numeric) < tol, f"{analytic[idx]} vs {numeric}"


def test_matmul_broadcast_grads():
    rng = np.random.default_rng(1)
    a = Tensor(rng.normal(size=(2, 3, 4)))
    w = Tensor(rng.normal(size=(4, 5)))

    def build():
        a.grad = np.zeros_like(a.data)
        w.grad = np.zeros_like(w.data)
        return a.matmul(w).tanh().sum()
    numeric_check(build, a)
    numeric_check(build, w)


def test_softmax_grads():
    rng = np.random.default_rng(2)
    x = Tensor(rng.normal(size=(3, 5)))

    def build():
        x.grad = np.zeros_like(x.data)
        return (x.softmax() * Tensor(np.arange(5.0))).sum()
    numeric_check(build, x)


def test_embedding_accumulates_repeated_rows():
    rng = np.random.default_rng(3)
    table = Tensor(rng.normal(size=(4, 3)))
    idx = np.array([1, 1, 2])  # row 1 used twice -> blame twice (the += rule)

    def build():
        table.grad = np.zeros_like(table.data)
        return embedding(table, idx).sum()
    numeric_check(build, table)


def test_cross_entropy_grads():
    rng = np.random.default_rng(4)
    logits = Tensor(rng.normal(size=(2, 3, 5)))
    targets = np.array([[0, 2, 4], [1, 1, 3]])

    def build():
        logits.grad = np.zeros_like(logits.data)
        return cross_entropy(logits, targets)
    numeric_check(build, logits)


def test_causal_mask_blocks_the_future():
    model = TinyGPT(vocab_size=7, block_size=6, d_model=8)
    idx = np.array([[1, 2, 3, 4, 5, 6]])
    before = model.forward(idx).data.copy()
    idx2 = idx.copy()
    idx2[0, 4] = 0  # change a LATER character
    after = model.forward(idx2).data
    # positions 0-3 come before the change and must be untouched
    assert np.allclose(before[0, :4], after[0, :4])
    assert not np.allclose(before[0, 4:], after[0, 4:])


def test_training_reduces_surprise():
    rng = np.random.default_rng(5)
    model = TinyGPT(vocab_size=5, block_size=4, d_model=8, rng=rng)
    opt = Adam(model.params(), lr=0.01)
    x = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    y = np.array([[1, 2, 3, 4], [2, 3, 4, 0]])
    first = model.loss(x, y).data
    for _ in range(60):
        loss = model.loss(x, y)
        model.zero_grad()
        loss.backward()
        opt.step()
    assert loss.data < first * 0.5


def test_sampling_is_valid():
    model = TinyGPT(vocab_size=4, block_size=4, d_model=8)
    encode = {".": 0, "a": 1, "b": 2, "c": 3}
    decode = {i: c for c, i in encode.items()}
    word = sample_word(model, encode, decode, np.random.default_rng(6), max_len=15)
    assert len(word) <= 15
    assert all(ch in "abc" for ch in word)
