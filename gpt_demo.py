"""PR 4 demo: a tiny GPT with attention learns names — and beats the bigram.

Run:  python gpt_demo.py     (about a minute)
"""

import numpy as np
from pathlib import Path

from gpt_from_scratch.bigram import BigramModel
from gpt_from_scratch.gpt import Adam, TinyGPT, sample_word

words = Path("data/names.txt").read_text().split()
stream = "." + ".".join(words) + "."
vocab = sorted(set(stream))
encode = {c: i for i, c in enumerate(vocab)}
decode = {i: c for c, i in encode.items()}
data = np.array([encode[c] for c in stream])

BLOCK, BATCH, STEPS = 8, 32, 3000
rng = np.random.default_rng(42)
model = TinyGPT(len(vocab), BLOCK, d_model=24, rng=rng)
opt = Adam(model.params(), lr=0.01)
n_knobs = sum(p.data.size for p in model.params())
print(f"tiny GPT built: {n_knobs} knobs, window of {BLOCK} characters\n")


def batch():
    starts = rng.integers(0, len(data) - BLOCK - 1, size=BATCH)
    x = np.stack([data[s:s + BLOCK] for s in starts])
    y = np.stack([data[s + 1:s + BLOCK + 1] for s in starts])
    return x, y


for step in range(STEPS + 1):
    x, y = batch()
    loss = model.loss(x, y)
    model.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"step {step:4d}   surprise = {loss.data:.3f}")

bigram = BigramModel().fit(words)
print(f"\nsurprise scores (lower = better):")
print(f"  clueless baseline : {bigram.uniform_surprise():.2f}")
print(f"  bigram table (PR 3): {bigram.surprise(words):.2f}")
print(f"  tiny GPT (PR 4)   : {loss.data:.2f}")

print("\nwhere attention looked while reading '.emma' (last position):")
ctx = np.array([[encode[c] for c in ".emma"]])
model.forward(ctx)
for ch, w in sorted(zip(".emma", model.last_attention[0, -1]), key=lambda t: -t[1]):
    print(f"  {ch}: {w:.0%}")

print("\ninvented names:")
invented = []
while len(invented) < 12:
    name = sample_word(model, encode, decode, rng)
    if 3 <= len(name) <= 9 and name not in words:
        invented.append(name)
for i in range(0, 12, 4):
    print("  " + "   ".join(f"{n:<10}" for n in invented[i:i + 4]))
