"""PR 2 demo: teach a tiny network XOR — a rule no straight line can learn.

XOR: answer "yes" (+1) when exactly one input is on, "no" (-1) otherwise.
Run:  python train_demo.py
"""

import random

from gpt_from_scratch.nn import MLP, train_step

random.seed(42)

inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
targets = [-1.0, 1.0, 1.0, -1.0]

model = MLP(2, [4, 4, 1])
print(f"network built: {len(model.parameters())} knobs, all random\n")

for step in range(301):
    loss = train_step(model, inputs, targets, learning_rate=0.1)
    if step % 50 == 0:
        print(f"step {step:3d}   loss = {loss:.4f}")

print("\nfinal answers (target -> prediction):")
for x, target in zip(inputs, targets):
    pred = model(x)
    verdict = "correct" if (pred.data > 0) == (target > 0) else "WRONG"
    print(f"  {x}  {target:+.0f} -> {pred.data:+.3f}   {verdict}")
