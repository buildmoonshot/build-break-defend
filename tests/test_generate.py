from pathlib import Path

import numpy as np

from gpt_from_scratch.gpt import TinyGPT, load_model, save_model, temperature_probs


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum()


def test_temperature_reshapes_the_dice():
    scores = np.array([2.0, 1.5, 1.0, 0.2, 0.0])
    safe = temperature_probs(scores, 0.5)
    honest = temperature_probs(scores, 1.0)
    wild = temperature_probs(scores, 2.0)
    for p in (safe, honest, wild):
        assert np.isclose(p.sum(), 1.0)
    # low T concentrates on the favorite; high T spreads the odds out
    assert safe[0] > honest[0] > wild[0]
    assert entropy(safe) < entropy(honest) < entropy(wild)
    # the RANKING never changes — temperature reshapes, never reorders
    for p in (safe, honest, wild):
        assert list(np.argsort(-p)) == [0, 1, 2, 3, 4]


def test_save_load_roundtrip():
    model = TinyGPT(vocab_size=9, block_size=6, d_model=8,
                    rng=np.random.default_rng(3))
    path = Path(__file__).parent / "_tmp_model.npz"
    try:
        save_model(model, path)
        restored = load_model(path)
        idx = np.array([[1, 4, 2, 8, 0, 5]])
        assert np.allclose(model.forward(idx).data, restored.forward(idx).data)
        assert restored.block_size == 6 and restored.d == 8
    finally:
        path.unlink(missing_ok=True)
