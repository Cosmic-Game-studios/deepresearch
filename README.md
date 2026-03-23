# DeepResearch 🔬

**Autonomous experiment loops that beat greedy search. Proven.**

> _Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Same core loop. Smarter strategy. Better results._

![DeepResearch vs Autoresearch convergence](progress.png)

## Why DeepResearch?

Autoresearch is brilliant — an AI agent modifies code, tests it, keeps wins, reverts losses, and repeats overnight. But it uses **greedy hill-climbing**: if a change doesn't immediately improve the metric, it's discarded. This gets stuck in local optima.

DeepResearch replaces random exploration with **intelligent search**:

| What | Autoresearch | DeepResearch |
|---|---|---|
| How it thinks | Doesn't — blind search | **Reasoning Layer** — reads code, forms theories, reflects |
| Category selection | Random | **Thompson Sampling** — learns which change types work |
| Worse results | Always discard | **Simulated Annealing** — accept worse early to escape traps |
| Branches | 1 best | **Population of K** — parallel exploration + crossover |
| Between sessions | Forget everything | **Persistent memory** — patterns, anti-patterns, causal deps |
| Learning | Count successes | **Causal models** — understands WHY things work |
| After 50 experiments | Barely moving | **+28.4% better** (proven) |

## Prove It

### The Reasoning Layer is what matters

![Reasoning Layer proof](reasoning_proof.png)

The mechanical strategy (bandit + annealing + population) without understanding is **worse than greedy**. The intelligence comes from THINKING, not from the algorithm:

```bash
python benchmark_reasoning.py
```

```
  GREEDY (autoresearch)            mean= 68.9     (baseline)
  DR Mechanical (no reasoning)     mean= 72.2     (-4.8% — worse!)
  DR + Reasoning Layer             mean= 62.2     (+9.8% — the only winner)

  ★ Value of Reasoning Layer alone: +13.9%
```

### It scales — biggest advantage when experiments are expensive

| Experiments | Greedy | DR + Reasoning | Improvement |
|---|---|---|---|
| 50 | 101.5 | 72.7 | **+28.4%** |
| 100 | 78.2 | 63.2 | **+19.2%** |
| 200 | 68.9 | 62.2 | **+9.8%** |
| 500 | 66.3 | 61.1 | **+7.9%** |

The advantage is largest at low experiment counts — exactly where it matters most (experiments are expensive).

### Head-to-head convergence

![DeepResearch vs Autoresearch convergence](progress.png)

```bash
python compare.py          # generates progress.png
python compare.py --scaling # scaling table
```

## How It Works

Other tools optimize the mechanical loop. DeepResearch optimizes the **thinking** that drives the loop.

```
Every experiment:
  1. DEEP READ    — Read and understand the artifact (not just grep for numbers)
  2. THEORIZE     — Form causal hypothesis: "metric limited by X because Y"
  3. PREDICT      — "Changing Z should improve by ~N% because [mechanism]"
  4. MUTATE       — One targeted change to test the theory
  5. EXECUTE      — Fixed budget eval
  6. REFLECT      — Was prediction correct? WHY? Update mental model
  7. LOG          — Hypothesis + result + learning (not just "kept/reverted")

Every 10 experiments:
  → Research memo: theory, findings, dead ends, next direction

The model's intelligence IS the search strategy.
```

Built for Opus 4.6: adaptive thinking for deep reasoning, 1M context for holding entire research histories, interleaved thinking for reasoning between tool calls.

## Quick Start

### As a Claude Code skill

```bash
# Point your agent at the SKILL.md:
# "Read SKILL.md and start deepresearch on my project"
```

### Standalone

```bash
# 1. Initialize
bash init.sh --domain ml     # or: code, prompt, game, doc

# 2. Configure
vim .deepresearch/config.json  # Set target_files, metric

# 3. Create eval harness (see SKILL.md for templates)
vim .deepresearch/eval.sh

# 4. Run
# Tell your AI agent: "Read SKILL.md and run deepresearch"
```

## Works on Everything

DeepResearch is domain-agnostic. If you can measure it, you can optimize it:

- **ML training** — val_bpb, accuracy, loss (like autoresearch)
- **Code performance** — benchmark time, memory usage, test pass rate
- **Prompt engineering** — LLM-as-judge scores, task accuracy
- **Game balancing** — fairness index, win rate variance
- **Document quality** — rubric scores, readability metrics
- **Config tuning** — throughput, latency, resource usage

## What's in the Box

```
compare.py              ← Run this first. Head-to-head convergence chart.
benchmark_reasoning.py  ← THE proof. Reasoning Layer vs blind search.
SKILL.md                ← Full agent instructions with Reasoning Layer protocol
reasoning_layer.md      ← Deep dive: R1 (Deep Read), R2 (Hypothesis), R3 (Reflection)
strategy.py             ← Thompson Sampling + annealing + population engine
init.sh                 ← One-command project setup
templates/
  research.md           ← Human-facing research goals template
```

## The Reasoning Layer (the core differentiator)

Every other autoresearch tool optimizes the mechanical loop. We optimize the thinking.

**R1: Deep Read** — Before mutating, read and understand the artifact. Identify the bottleneck. Not "what could change" but "what limits the metric RIGHT NOW."

**R2: Causal Hypothesis** — Don't pick random mutations. Form a theory: "The metric is X because of Y. Changing Z should improve it because [mechanism]. This connects to experiment #N which showed [finding]."

**R3: Reflection** — After seeing results, don't just update a counter. Ask: was my prediction correct? Why? Has the bottleneck shifted? Write a 2-3 sentence reflection that updates your mental model.

**Research Memos** — Every 10 experiments, synthesize findings into a memo: current theory, key learnings, dead ends, next direction. These compound across the session — memo #5 references #3 which references #1.

**Causal Dependencies** — Track which changes depend on each other. Enables smart ablation and smart crossover.

See `reasoning_layer.md` for the full protocol.

## The Strategy Engine (tools the researcher uses)

**Thompson Sampling** for category selection:
- Each mutation category (architecture, optimizer, etc.) is a bandit arm
- Beta(α, β) distribution tracks successes/failures
- Agent naturally focuses on categories that work
- Forced exploration with probability proportional to temperature

**Simulated Annealing** for escaping local optima:
- Early experiments accept small regressions (high temperature)
- Late experiments are nearly greedy (low temperature)
- Adaptive reheat when stuck for 8+ experiments

**Population Search** for diversity:
- K branches explore different directions simultaneously
- Crossover at phase transition: combine best traits
- Tournament selection favors better branches

**Persistent Memory** across sessions:
- `.deepresearch/knowledge.json` stores patterns and anti-patterns
- Session 2 starts smarter than session 1 ended
- Never repeats known-bad approaches

## Contributing

The benchmark is the source of truth. Any improvement must show up in `python compare.py`. PRs welcome.

## License

MIT
