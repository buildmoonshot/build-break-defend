import math
import random

from gpt_from_scratch.bigram import BigramModel

WORDS = ["emma", "mia", "amelia", "liam", "emil", "mila"]


def test_rows_are_probability_distributions():
    model = BigramModel().fit(WORDS)
    for row in model.probs:
        assert math.isclose(sum(row), 1.0, rel_tol=1e-9)
        assert all(p > 0 for p in row)  # smoothing: nothing is impossible


def test_learns_the_data():
    model = BigramModel().fit(WORDS)
    # 'm' is followed by vowels constantly in this data, never by 'l'
    assert model.prob("m", "a") > model.prob("m", "l")


def test_beats_clueless_baseline():
    model = BigramModel().fit(WORDS)
    assert model.surprise(WORDS) < model.uniform_surprise()


def test_sampling_is_reproducible_and_valid():
    model = BigramModel().fit(WORDS)
    a = model.sample_word(random.Random(7))
    b = model.sample_word(random.Random(7))
    assert a == b  # same seed, same dice, same word
    assert all(ch in model.vocab for ch in a)
    assert BigramModel.BOUNDARY not in a


def test_sampling_terminates():
    model = BigramModel().fit(WORDS)
    rng = random.Random(1)
    for _ in range(200):
        assert len(model.sample_word(rng)) <= 30
