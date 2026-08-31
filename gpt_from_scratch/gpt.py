"""A tiny GPT: attention bolted onto next-character guessing (PR 4).

The bigram model (PR 3) remembered one character back. This model reads
a window of characters and, at every position, LOOKS BACK over everything
earlier, weighing which characters matter right now — that weighted
glance-back is attention. The weights aren't hardcoded; they're learned
by the same blame-and-nudge loop as PR 2.

Each position broadcasts a question (query: "who has info I need?") and
a resume (key: "here's what I am"); where they match, attention flows,
and the matched positions contribute what they know (value).

See docs/pr4-attention.svg for the picture version.
"""

import math

import numpy as np

from gpt_from_scratch.tensor import Tensor, cross_entropy, embedding


class TinyGPT:
    def __init__(self, vocab_size, block_size, d_model=24, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)

        def weights(*shape, scale):
            return Tensor(rng.normal(0.0, scale, size=shape))

        s = 1.0 / math.sqrt(d_model)
        self.block_size = block_size
        self.d = d_model
        self.tok_emb = weights(vocab_size, d_model, scale=0.1)   # what each char IS
        self.pos_emb = weights(block_size, d_model, scale=0.1)   # WHERE it sits
        self.Wq = weights(d_model, d_model, scale=s)             # makes questions
        self.Wk = weights(d_model, d_model, scale=s)             # makes resumes
        self.Wv = weights(d_model, d_model, scale=s)             # makes contributions
        self.Wo = weights(d_model, d_model, scale=s)
        self.W1 = weights(d_model, 4 * d_model, scale=s)         # think it over
        self.W2 = weights(4 * d_model, d_model, scale=1.0 / math.sqrt(4 * d_model))
        self.head = weights(d_model, vocab_size, scale=s)        # scores per char
        # causal mask: a position may only look at itself and earlier — never
        # the future (that would be reading the answer key)
        self._mask = np.triu(np.full((block_size, block_size), -1e9), k=1)
        self.last_attention = None

    def params(self):
        return [self.tok_emb, self.pos_emb, self.Wq, self.Wk, self.Wv,
                self.Wo, self.W1, self.W2, self.head]

    def forward(self, idx):
        idx = np.asarray(idx)                                   # (batch, T)
        T = idx.shape[-1]
        x = embedding(self.tok_emb, idx) + embedding(self.pos_emb, np.arange(T))
        # attention: match questions to resumes, blend contributions
        q = x.matmul(self.Wq)
        k = x.matmul(self.Wk)
        v = x.matmul(self.Wv)
        scores = q.matmul(k.transpose_last()) * (1.0 / math.sqrt(self.d))
        scores = scores + Tensor(self._mask[:T, :T])
        attn = scores.softmax()                                 # the look-back weights
        self.last_attention = attn.data
        x = x + attn.matmul(v).matmul(self.Wo)                  # blend joins the stream
        # a small "think it over" layer, then scores for every character
        x = x + x.matmul(self.W1).tanh().matmul(self.W2)
        return x.matmul(self.head)                              # (batch, T, vocab)

    def loss(self, idx, targets):
        return cross_entropy(self.forward(idx), targets)

    def zero_grad(self):
        for p in self.params():
            p.grad = np.zeros_like(p.data)


class Adam:
    """A smarter nudger: a per-knob auto-adjusting shower handle.

    Plain SGD uses one nudge size for all knobs. Adam watches each knob's
    recent blame and shrinks the step for twitchy knobs, grows it for
    sleepy ones — everyone settles toward their valley at their own pace.
    """

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = 0.9 * self.m[i] + 0.1 * p.grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * p.grad ** 2
            m_hat = self.m[i] / (1 - 0.9 ** self.t)
            v_hat = self.v[i] / (1 - 0.999 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def temperature_probs(scores, temperature=1.0):
    """Turn raw scores into dice, with a boldness dial (PR 5).

    Dividing scores by T rescales the GAPS between them before they
    become odds: small T stretches gaps (the favorite dominates), big T
    shrinks them (underdogs get real chances). T=1 leaves them honest.
    """
    z = scores / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def sample_word(model, encode, decode, rng, boundary=".", max_len=30, temperature=1.0):
    """Invent a word: same dice-rolling walk as PR 3, smarter table."""
    context = [encode[boundary]]
    out = []
    while len(out) < max_len:
        window = context[-model.block_size:]
        logits = model.forward(np.array([window]))
        probs = temperature_probs(logits.data[0, -1], temperature)
        choice = rng.choice(len(probs), p=probs)
        if choice == encode[boundary]:
            break
        out.append(decode[choice])
        context.append(choice)
    return "".join(out)


_PARAM_NAMES = ["tok_emb", "pos_emb", "Wq", "Wk", "Wv", "Wo", "W1", "W2", "head"]


def save_model(model, path):
    """A trained model IS its knob values — the whole thing fits in one file."""
    arrays = {name: getattr(model, name).data for name in _PARAM_NAMES}
    config = np.array([model.tok_emb.data.shape[0], model.block_size, model.d])
    np.savez(path, __config__=config, **arrays)


def load_model(path):
    stored = np.load(path)
    vocab_size, block_size, d_model = (int(v) for v in stored["__config__"])
    model = TinyGPT(vocab_size, block_size, d_model=d_model)
    for name in _PARAM_NAMES:
        getattr(model, name).data[:] = stored[name]
    return model
