"""Character-level bigram language model (PR 3).

A language model has one job: guess what comes next. This is the
simplest possible one — a table with a row for every character, where
each row says how often each other character follows it. Counting the
pairs in the training words IS the training: for a model this simple,
the tally is provably the best table there is. Gradient descent comes
back in PR 4, when the model outgrows being a table.

See docs/pr3-bigram.svg for the picture version.
"""

import math


class BigramModel:
    BOUNDARY = "."  # marks the edge of a word, so the model learns starts and ends

    def __init__(self, smoothing=0.1):
        # pretend every pair was seen `smoothing` extra times, so nothing
        # real ever gets probability zero (zero would mean infinite surprise).
        # keep it small: with a small dataset, big smoothing drowns the real
        # counts in uniform noise and the samples turn to junk
        self.smoothing = smoothing

    def fit(self, words):
        chars = sorted(set("".join(words)))
        self.vocab = [self.BOUNDARY] + chars
        self.index = {c: i for i, c in enumerate(self.vocab)}
        n = len(self.vocab)
        counts = [[self.smoothing] * n for _ in range(n)]
        for word in words:
            seq = self.BOUNDARY + word + self.BOUNDARY
            for a, b in zip(seq, seq[1:]):
                counts[self.index[a]][self.index[b]] += 1
        # turn each row of counts into a row of probabilities (sums to 1)
        self.probs = [[c / sum(row) for c in row] for row in counts]
        return self

    def prob(self, a, b):
        """Probability that character b comes right after character a."""
        return self.probs[self.index[a]][self.index[b]]

    def next_chars(self, a, top=5):
        """The most likely characters to follow a, best first."""
        row = self.probs[self.index[a]]
        ranked = sorted(zip(self.vocab, row), key=lambda kv: -kv[1])
        return ranked[:top]

    def sample_word(self, rng, max_len=30):
        """Invent a word by walking the table: roll weighted dice, repeat."""
        out = []
        current = self.BOUNDARY
        while len(out) < max_len:
            row = self.probs[self.index[current]]
            roll, acc, nxt = rng.random(), 0.0, self.vocab[-1]
            for ch, p in zip(self.vocab, row):
                acc += p
                if roll <= acc:
                    nxt = ch
                    break
            if nxt == self.BOUNDARY:
                break
            out.append(nxt)
            current = nxt
        return "".join(out)

    def surprise(self, words):
        """Average surprise per character pair (negative log likelihood).

        A model guessing blindly over V characters scores log(V).
        Lower = the model genuinely expects what the data does.
        """
        total, count = 0.0, 0
        for word in words:
            seq = self.BOUNDARY + word + self.BOUNDARY
            for a, b in zip(seq, seq[1:]):
                total += -math.log(self.prob(a, b))
                count += 1
        return total / count

    def uniform_surprise(self):
        """The clueless baseline: equal odds on every character."""
        return math.log(len(self.vocab))
