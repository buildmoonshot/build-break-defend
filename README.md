# build-break-defend

Learning AI by building it, in public, in three phases:

1. **Build** — a small GPT from scratch, so language models stop being magic.
2. **Break-worthy** — a real security tool (an LLM phishing-email classifier) with an
   honest eval against a dumb baseline.
3. **Defend** — attack my own tool with prompt-injection techniques, then defend it
   and measure the difference.

Every step ships as a small pull request with working, tested code and a diagram
explaining the concept. No real job data ever appears in this repo — datasets are
public or synthetic.

## Roadmap

### Phase 1 — build a small GPT from scratch

| PR | Builds | Teaches |
|---|---|---|
| 1 | Tiny autograd engine (~100 lines) + tests | What a gradient is |
| 2 | Small neural net on toy data, using the PR-1 engine | The training loop and loss |
| 3 | Character-level bigram language model | How text becomes numbers and probabilities |
| 4 | Attention + a tiny transformer trained on a small corpus | How GPT actually works |
| 5 | Text generation + a plain-English write-up | Explaining it simply |

### Phase 2 — a real tool with an honest eval

| PR | Builds | Teaches |
|---|---|---|
| 6 | Phishing dataset + dumb baseline (keyword/regex) + eval harness | Evals before models |
| 7 | LLM phishing classifier scored on the same eval | Where AI beats a for-loop, measured |
| 8 | Error analysis + prompt iteration | Where LLMs fail and why |
| 9 | 30-second demo: CLI + README demo | If it doesn't run, it doesn't count |

### Phase 3 — security of LLM apps

| PR | Builds | Teaches |
|---|---|---|
| 10 | Prompt-injection attack suite vs. my own PR-7 classifier | OWASP LLM Top 10, adversarial inputs |
| 11 | Defenses + re-run of the attack suite, before/after numbers | Defense in depth for LLM apps |
| 12 | Capstone write-up: attacking and defending my own tool | The security × AI intersection |

## Setup

```
make setup    # create venv and install dependencies
make test     # run the test suite
```

Without `make`: `python -m venv .venv`, activate it, `pip install -r requirements.txt`,
then `pytest`.

## Layout

```
docs/    design notes and the diagrams that explain each PR
```

Source directories are added by the PR that first needs them — nothing speculative.

## License

MIT — see [LICENSE](LICENSE).
