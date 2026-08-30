"""PR 3 demo: a model reads 445 names, learns English spelling habits,
and invents names nobody has ever had.

Run:  python bigram_demo.py
"""

import random
from pathlib import Path

from gpt_from_scratch.bigram import BigramModel

words = Path("data/names.txt").read_text().split()
model = BigramModel().fit(words)

print(f"read {len(words)} names, vocabulary of {len(model.vocab)} characters\n")

print("what the model learned about English spelling:")
for ch in ["q", "e", "."]:
    top = ", ".join(f"{c} {p:.0%}" for c, p in model.next_chars(ch, top=3))
    label = "a name starts with" if ch == "." else f"after '{ch}' comes"
    print(f"  {label}:  {top}")

print(f"\nsurprise score (lower = better):")
print(f"  clueless baseline : {model.uniform_surprise():.2f}")
print(f"  trained model     : {model.surprise(words):.2f}")

rng = random.Random(42)
print("\ninvented names (never seen in the data):")
invented = []
while len(invented) < 12:
    name = model.sample_word(rng)
    if 3 <= len(name) <= 9 and name not in words:
        invented.append(name)
for i in range(0, 12, 4):
    print("  " + "   ".join(f"{n:<10}" for n in invented[i:i + 4]))
