"""PR 5: the name generator, as a real command.

First run trains the tiny GPT (~1 minute) and saves it; after that,
generation is instant — because a trained model is nothing but its
8,400 knob values, and those fit in one small file.

Run:  python generate.py
      python generate.py --count 20 --temperature 0.6
      python generate.py --temperature 1.5
      python generate.py --retrain
"""

import argparse
from pathlib import Path

import numpy as np

from gpt_from_scratch.gpt import Adam, TinyGPT, load_model, sample_word, save_model

MODEL_PATH = Path("models/tiny_gpt.npz")
BLOCK, BATCH, STEPS = 8, 32, 3000

parser = argparse.ArgumentParser(description="Invent names with the tiny GPT.")
parser.add_argument("--count", type=int, default=12, help="how many names")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="boldness dial: 0.5 = safe and samey, 2.0 = wild")
parser.add_argument("--seed", type=int, default=None, help="dice seed (default: random)")
parser.add_argument("--retrain", action="store_true", help="train fresh even if a saved model exists")
args = parser.parse_args()

words = Path("data/names.txt").read_text().split()
stream = "." + ".".join(words) + "."
vocab = sorted(set(stream))
encode = {c: i for i, c in enumerate(vocab)}
decode = {i: c for c, i in encode.items()}

if MODEL_PATH.exists() and not args.retrain:
    model = load_model(MODEL_PATH)
    print(f"loaded trained model from {MODEL_PATH}")
else:
    print("no saved model — training (about a minute)...")
    data = np.array([encode[c] for c in stream])
    rng = np.random.default_rng(42)
    model = TinyGPT(len(vocab), BLOCK, d_model=24, rng=rng)
    opt = Adam(model.params(), lr=0.01)
    for step in range(STEPS + 1):
        starts = rng.integers(0, len(data) - BLOCK - 1, size=BATCH)
        x = np.stack([data[s:s + BLOCK] for s in starts])
        y = np.stack([data[s + 1:s + BLOCK + 1] for s in starts])
        loss = model.loss(x, y)
        model.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:4d}   surprise = {loss.data:.3f}")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    save_model(model, MODEL_PATH)
    print(f"saved to {MODEL_PATH}")

rng = np.random.default_rng(args.seed)
print(f"\n{args.count} invented names at temperature {args.temperature}:")
made = []
attempts = 0
while len(made) < args.count and attempts < args.count * 40:
    attempts += 1
    name = sample_word(model, encode, decode, rng, temperature=args.temperature)
    if 3 <= len(name) <= 10:
        made.append(name)
for i in range(0, len(made), 4):
    print("  " + "   ".join(f"{n:<11}" for n in made[i:i + 4]))
