---
name: deepresearch
description: >
  Advanced autonomous research engine inspired by Karpathy's autoresearch.
  Goes far beyond greedy hill-climbing with: Bayesian bandit exploration,
  simulated annealing, population-based search across parallel branches,
  persistent cross-session memory, multi-agent parallel experiments, and
  auto-generated research reports. Works on anything with a measurable metric:
  ML training, code optimization, prompt engineering, game balancing, config
  tuning, document quality, and more.
  Trigger phrases: "run deepresearch", "optimize this autonomously",
  "run experiments overnight", "iterate until better", "deep optimize",
  "research loop on this", "autoloop", "autoresearch", "hill-climb this",
  "run the Karpathy loop", "explore and exploit".
---

# DeepResearch — Autonomous Research Engine v3

Autoresearch tools treat LLMs as code editors following bandit statistics.
DeepResearch treats the LLM as a **thinking researcher** — one that reads
code, forms causal theories, designs experiments to test them, and reflects
on results to update its understanding. Statistics inform but don't dictate.

Built for Opus 4.6: leverages adaptive thinking for deep reasoning,
1M context for holding entire research histories, and interleaved thinking
for reasoning between tool calls. The model's intelligence IS the search
strategy.

Seven layers:

1. **Reasoning Layer** — Deep read → causal hypothesis → reflection (the core differentiator)
2. **Knowledge Acquisition** — External docs, papers, technique extraction, evidence-based hypotheses
3. **Strategy Engine** — Bayesian bandit + simulated annealing as researcher tools
4. **Generative Mutations** — Level 2 structural add/remove/replace with safety rails
5. **Curriculum System** — Progressive goals with stage-specific strategies
6. **Persistent Memory** — Knowledge base, patterns, anti-patterns, causal dependencies, techniques
7. **Autonomous Pipeline** — Level 3: spec → research → architect → build → optimize → report

**Navigation** (for agents parsing this file):
- **Phase 0** — Setup (config, eval harness templates, baseline)
- **Reasoning Layer** — Deep Read + Knowledge Acquisition, Hypothesis, Reflection, Memos
- **Level 2-3** — Generative Mutations, Curriculum, Domain Research Protocol, Architecture Planning, Multi-File Scope
- **Phase 1** — Core Loop (select → hypothesize → mutate → execute → score → log)
- **Phase 1 Walkthrough** — End-to-end example with exact commands
- **Phase 2** — Strategy Engine (momentum, plateau detection, regression, restarts)
- **Phase 3** — Persistent Memory (knowledge.json schema, update protocol)
- **Phase 4** — Parallel Experiments (git worktrees, multi-GPU)
- **Phase 5** — Research Reports (templates, auto-generation triggers)
- **Domain Configurations** — ML, Code, Prompt, Game, Document presets
- **Stopping Conditions** — When and how to end
- **Error Recovery** — Validation, corruption repair, backups, safety
- **Self-Improvement** — Meta-optimization config, eval criteria, safety rails

The human's job: define what "better" means, set constraints, write `research.md`.
The agent's job: everything else.

---

## Directory Structure

DeepResearch uses a `.deepresearch/` directory at the project root for all
persistent state. This directory is THE brain — it persists across sessions.

```
.deepresearch/
├── config.json          # Session config (metric, target, budget, etc.)
├── knowledge.json       # Cross-session knowledge base (patterns, anti-patterns)
├── dependencies.json    # Causal dependency graph between experiments
├── experiments.jsonl     # Append-only experiment log (one JSON per line)
├── curriculum.json      # Progressive goals (Level 2.5)
├── orchestrator_state.json  # Level 3 pipeline state
├── architecture_plan.json   # Level 3 component plan
├── research/            # External domain knowledge (Level 1.5+)
│   ├── sources.json     # Reading list: URLs, summaries, insights
│   ├── techniques.json  # Extracted techniques with evidence + priority
│   └── domain_knowledge.json  # Structured research findings
├── memos/               # Research memos (every 10 experiments)
│   └── memo-10.md       # Synthesized findings and theories
├── populations/          # Top-K branch snapshots
│   ├── branch-0/         # Baseline snapshot
│   ├── branch-1/         # Best variant 1
│   └── branch-2/         # Best variant 2
├── reports/              # Auto-generated research reports
│   └── session-YYYYMMDD-HHMM.md
└── strategy-state.json   # Bandit arms, temperature, scores
```

On first run, create this structure. On subsequent runs, LOAD it and continue
where the last session left off. This is the key differentiator from vanilla
autoresearch — DeepResearch has memory.

---

## Phase 0 — Setup

### 0.1 Define the Research Problem

Work with the human to establish these BEFORE the loop starts:

**Target artifact(s):** What file(s) get modified? Can be one file (like
a script or config) or a small set of related files. Fewer is better.

**Primary metric:** ONE number. Direction must be clear (lower/higher = better).
Examples: response latency ms (lower), test pass rate (higher), benchmark score
(higher), LLM judge score 0-10 (higher), val_loss (lower), fairness index (higher).

**Evaluation harness:** How to compute the metric. This is IMMUTABLE during the
loop. Can be: a script, a test suite, an LLM-as-judge prompt, a benchmark.
Write it to `.deepresearch/eval.sh` or `.deepresearch/eval.py`.

Use one of these ready-made templates:

**Template A — Script metric (any script that outputs a number):**
```bash
#!/bin/bash
# .deepresearch/eval.sh — extract a single number from a script run
set -e
BUDGET="${1:-300}"
TARGET="${2:-app.py}"
LOG=".deepresearch/run.log"
timeout "${BUDGET}s" python "$TARGET" > "$LOG" 2>&1 || true
METRIC=$(grep "^metric:\|^score:\|^result:" "$LOG" | tail -1 | awk '{print $2}')
if [ -z "$METRIC" ]; then
  echo "metric: CRASHED"
  tail -50 "$LOG"
  exit 1
fi
echo "metric: $METRIC"
```

**Template B — LLM-as-judge (prompt engineering, document quality):**
```bash
#!/bin/bash
# .deepresearch/eval.sh — score an artifact using Claude as judge with rubric
set -e
ARTIFACT="${1:?Usage: eval.sh <artifact_path>}"
CONTENT=$(cat "$ARTIFACT")

# IMPORTANT: Define your rubric ONCE and never change it during the loop.
# Each criterion is binary (0 or 1). Total = sum / N * 10.
JUDGE_PROMPT="Score this artifact on these criteria (0=fail, 1=pass each).
Respond ONLY with JSON: {\"c1\":0or1, \"c2\":0or1, ..., \"total\":N}

CRITERIA:
1. CLEAR: Instructions are unambiguous and directly actionable
2. COMPLETE: All necessary steps are covered, no gaps
3. CORRECT: No factual errors or logical contradictions
4. CONCISE: No redundancy, every sentence earns its place
5. ACTIONABLE: Contains copy-paste-ready commands or code

ARTIFACT:
$CONTENT"

RESPONSE=$(curl -s https://api.anthropic.com/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d "$(jq -n --arg p "$JUDGE_PROMPT" '{
    model: "claude-sonnet-4-6",
    max_tokens: 200,
    messages: [{role: "user", content: $p}]
  }')")
TOTAL=$(echo "$RESPONSE" | jq -r '.content[0].text' | jq -r '.total')
echo "metric: $TOTAL"
```

**Template C — Test suite (code optimization):**
```bash
#!/bin/bash
# .deepresearch/eval.sh — run tests and measure pass rate + performance
set -e
RESULTS=$(python -m pytest tests/ --tb=no -q 2>&1)
PASSED=$(echo "$RESULTS" | grep -oP '\d+(?= passed)')
TOTAL=$(echo "$RESULTS" | grep -oP '\d+(?= passed)' | head -1)
# Benchmark
START=$(date +%s%N)
python benchmark.py > /dev/null 2>&1
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
echo "metric: $MS"
echo "tests_passed: $PASSED"
```

**Template D — Composite metric (game balancing, multi-objective):**
```bash
#!/bin/bash
# .deepresearch/eval.sh — combine multiple sub-metrics into one score
set -e
# Run your simulation / test / benchmark
python simulate.py --config balance.json > /tmp/sim-results.json

# Extract sub-metrics and compute weighted composite
python3 -c "
import json
r = json.load(open('/tmp/sim-results.json'))
# Define weights (MUST be fixed — never change during the loop)
W_FAIRNESS = 0.4   # win rate variance (lower is better, so invert)
W_ENGAGEMENT = 0.3  # average session length (higher is better)
W_BALANCE = 0.3     # unit diversity in top strategies (higher is better)

# Normalize each to 0-1 range using known bounds
fairness = 1.0 - min(r['win_rate_variance'] / 0.25, 1.0)  # 0.25 = worst case
engagement = min(r['avg_session_minutes'] / 60.0, 1.0)      # 60 min = perfect
balance = min(r['unit_diversity_index'] / 1.0, 1.0)          # 1.0 = perfect

composite = (W_FAIRNESS * fairness + W_ENGAGEMENT * engagement + W_BALANCE * balance) * 10
print(f'metric: {composite:.2f}')
print(f'sub_fairness: {fairness:.3f}')
print(f'sub_engagement: {engagement:.3f}')
print(f'sub_balance: {balance:.3f}')
"
```

**Budget per experiment:** Fixed time or resource budget. Every experiment gets
exactly the same budget so results are comparable. Default: 5 minutes.

**Population size K:** How many parallel branches to maintain.

| Scenario | K | Why |
|---|---|---|
| Single GPU ML training | 1 | Can't parallelize, branches are overhead |
| Fast eval (<30s), single machine | 3 | Sweet spot: exploit + explore + wildcard |
| Fast eval + many mutation categories | 5 | More categories need more parallel search |
| Multi-GPU (4+ GPUs) | = GPU count | One branch per GPU, all run simultaneously |
| Very simple problem (few knobs) | 1 | No need for population diversity |
| Huge search space (10+ categories) | 5 | Maximum diversity before coordination overhead dominates |

**Temperature schedule:** Controls exploration vs exploitation. Choose one:
`"aggressive"` (large search space), `"moderate"` (default), `"conservative"`
(already well-optimized). See Temperature Schedule in Phase 1 for exact formulas.

**Mutation categories:** Domain-specific list of change types the agent can try.
See Domain Configurations below. Each category becomes a "bandit arm."

### 0.2 Initialize

```bash
mkdir -p .deepresearch/populations/branch-0 .deepresearch/reports

# Save config — adapt these values to your domain
cat > .deepresearch/config.json << 'EOF'
{
  "target_files": ["<YOUR_FILE>"],
  "metric": "<YOUR_METRIC>",
  "metric_direction": "lower",
  "budget_seconds": 300,
  "population_size": 3,
  "temperature_schedule": "moderate",
  "mutation_categories": ["<cat1>", "<cat2>", "<cat3>"],
  "created": "2026-01-01T00:00:00Z",
  "session_count": 0
}
EOF
# Examples by domain:
#   ML:    target=train.py, metric=val_bpb, direction=lower, cats=[architecture,hyperparameters,optimizer]
#   Code:  target=app.py, metric=benchmark_ms, direction=lower, cats=[algorithm,memory,parallelism]
#   Prompt: target=prompt.md, metric=judge_score, direction=higher, cats=[structure,examples,tone]
#   Game:  target=balance.json, metric=fairness_idx, direction=higher, cats=[economy,combat,progression]

# Initialize strategy state
cat > .deepresearch/strategy-state.json << 'EOF'
{
  "temperature": 0.5,
  "total_experiments": 0,
  "bandit_arms": {},
  "population": [],
  "best_metric": null,
  "baseline_metric": null
}
EOF

# Initialize knowledge base (or load existing)
[ -f .deepresearch/knowledge.json ] || echo '{"patterns":[],"domain_insights":[],"anti_patterns":[]}' > .deepresearch/knowledge.json
```

### 0.3 Establish Baseline

Run the evaluation harness on the UNMODIFIED artifact. This is experiment #0.

```bash
# Branch for baseline
git checkout -b deepresearch/session-$(date +%Y%m%d)

# Run eval
bash .deepresearch/eval.sh > .deepresearch/run.log 2>&1

# Extract metric
BASELINE=$(grep "^metric:" .deepresearch/run.log | awk '{print $2}')

# Record
echo '{"id":0,"timestamp":"'$(date -Iseconds)'","branch":"baseline","category":"baseline","hypothesis":"Unmodified baseline","metric":'$BASELINE',"status":"baseline","description":"Initial baseline measurement"}' >> .deepresearch/experiments.jsonl

# Snapshot baseline into population
cp ${TARGET_FILES} .deepresearch/populations/branch-0/

# Update strategy state
# Set baseline_metric and best_metric to $BASELINE
```

Confirm baseline with the human. After confirmation: **NEVER ASK AGAIN. RUN AUTONOMOUSLY.**

---

## The Reasoning Layer — What Makes This Different

Every autoresearch tool treats the LLM as a code editor following bandit
statistics. DeepResearch treats it as a **thinking researcher** using
statistics as ONE input alongside understanding, theory, and reflection.

Opus 4.6 can reason about WHY experiments work, see patterns across 100+
experiments via its 1M context window, and form causal theories. The
Reasoning Layer wraps three thinking steps around every experiment.

**These thinking steps ARE the product.** The mechanical loop (mutate →
eval → keep/revert) is commodity — every tool does it. The thinking is
what makes experiments 28% more efficient at 50 experiments.

### R1: Deep Read (before mutating)

Don't grep for numbers. Build a mental model of the artifact.

**What to think about:**
1. Read the target file. What does each section/function/block do?
2. What is the **bottleneck** — the ONE thing limiting the metric most?
3. Has this bottleneck been targeted before? What happened?
4. What is my **causal model** of how the metric is produced?

**Example — good Deep Read (ML training):**
"The model is a 8-layer transformer with Muon+AdamW. Training runs for
5 minutes. Looking at the architecture: attention is standard full
attention — O(n²) in sequence length. With seq_len=1024 and 8 layers,
attention dominates compute. The optimizer has default hyperparams —
never tuned. I think the bottleneck is the optimizer, not the
architecture, because experiments #3 and #7 both changed architecture
with minimal effect, but experiment #5 (LR change) gave the biggest
improvement so far. Causal model: metric is limited by suboptimal LR
schedule → model doesn't converge fully in 5 minutes."

**Example — bad Deep Read:**
"The file has 200 lines. I'll change something in the architecture."

The difference: the good read identifies a SPECIFIC bottleneck with
evidence from past experiments. The bad read is just scanning.

**R1 Extended: External Knowledge Acquisition**

The Deep Read doesn't stop at the code. A good researcher also reads
external documentation, articles, and existing solutions BEFORE
experimenting. Use the knowledge acquisition system:

```python
from engine.knowledge import KnowledgeAcquisition

ka = KnowledgeAcquisition(domain="web_api", spec="Optimize REST API", language="python")

# 1. Generate targeted search queries
queries = ka.generate_searches(bottleneck="latency")
# → ["reducing web_api latency", "python web framework benchmarks", ...]

# 2. Search, read, register findings (agent uses web_search + web_fetch)
ka.register_source(url, title, source_type="documentation", relevance=0.9)
ka.mark_source_read(url, summary="...", key_insights=["insight 1", ...])

# 3. Extract techniques from what you read
ka.extract_technique(source_url=url, name="connection_pooling",
    description="Reuse DB connections across requests",
    expected_impact="30-50% latency reduction",
    evidence="Benchmarks show 3x throughput",
    applicable_when="DB calls are the bottleneck")

# 4. Get knowledge-backed hypothesis context for R2
context = ka.hypothesis_context("connection_pooling", "DB latency 142ms")
```

**When to do external research:**
- **First session on a new problem:** Read 5-7 sources before experimenting.
  Spend 2-3 min per source. Stop when you have 3+ techniques to try.
- **After hitting a plateau:** Search for techniques specific to your bottleneck.
- **Before a Level 2 structural mutation:** Read docs on the technique you're adding.
- **Never mid-experiment:** Research happens in R1, not during mutation.

**Reading protocol:** For each source, extract: techniques (most valuable),
architecture patterns, common pitfalls, and benchmarks. Call
`ka.reading_protocol()` for the full protocol.

The knowledge system persists in `.deepresearch/research/` — next session
picks up where you left off. Techniques you've tried and their results
carry forward, preventing repeated failures.

### R2: Causal Hypothesis (replaces "pick a random category")

The bandit INFORMS but doesn't DICTATE. Before making any change, write:

```
## Hypothesis #N
**Bottleneck:** [what limits the metric right now, from Deep Read]
**Theory:** [why this is the bottleneck — causal mechanism]
**Experiment:** Change [specific thing] to test this theory
**Prediction:** Metric should improve by ~[amount] because [mechanism]
**Connects to:** Builds on experiment #X which showed [finding]
**Risk:** Fails if [condition]. Fallback: [alternative]
**Confidence:** [low/medium/high] — low = speculative, high = strong evidence
```

**Example — good hypothesis:**
"Bottleneck: LR schedule. Theory: constant LR means the model overshoots
early and undershoots late, wasting training budget. Experiment: switch
to cosine schedule with 10% warmup. Prediction: ~2% improvement because
cosine is known to help with short training budgets. Connects to: exp #5
showed LR sensitivity — the landscape around the optimum is narrow.
Risk: warmup too long → wastes 10% of 5 min budget. Confidence: high."

**Example — bad hypothesis:**
"I'll try changing the learning rate. Bandit says hyperparameters work."

The good hypothesis has a THEORY of WHY. The bad one just follows statistics.

**Knowledge-backed hypothesis (Level 2+):**
When you have external knowledge from the acquisition system, reference it:

"Bottleneck: DB connection overhead (142ms per request, from R1 Deep Read).
Theory: Each request creates a new TCP connection + TLS handshake to the DB.
The FastAPI Deployment Guide (source #1) documents that connection pooling
reduces this to near-zero by reusing connections. Benchmarks in that source
show 3x throughput improvement. Experiment: Add sqlalchemy connection pool
with pool_size=20. Prediction: p99 drops from 142ms to ~60-80ms (50% reduction).
Connects to: exp #3 confirmed DB calls are the bottleneck (profiling data).
Risk: pool exhaustion under load → add max_overflow=10 as safety valve.
Confidence: high — both profiling data and external benchmarks agree."

Generate the context: `ka.hypothesis_context("connection_pooling", "DB latency")`

### R3: Reflection (after seeing results)

After EVERY experiment, answer these questions in 2-3 sentences:

1. **Prediction accuracy:** Was I right about direction AND magnitude?
2. **Why:** If correct → what's confirmed? If wrong → which of these:
   (a) wrong theory (b) right theory, wrong implementation (c) interaction effect
3. **Bottleneck shift:** Is the bottleneck still the same, or did this
   experiment move it somewhere else?
4. **Next move:** What is the LOGICAL next experiment given what I just learned?

**Example — good reflection:**
"Predicted ~2% improvement from cosine LR, got 1.8%. Theory confirmed:
short training budgets benefit from aggressive early LR. The bottleneck
has now shifted — LR is optimized, but the model is only 8 layers deep
and the loss curve shows it's still improving at t=5min. Next: try
depth=12 to give the model more capacity to USE the better LR schedule.
This is a dependent change — depth increase only makes sense because
we fixed the LR first."

**Example — bad reflection:**
"Metric improved by 1.8%. Kept."

The good reflection updates the causal model AND plans the next experiment.
The bad one logs nothing useful.

**If the experiment tested a technique from external research:**
Record the result so the knowledge system learns:
```python
ka.record_result("connection_pooling", "worked: p99 from 142ms to 85ms")
# or
ka.record_result("connection_pooling", "failed: incompatible with async framework")
```
Next time `ka.suggest_next()` is called, successful techniques inform
related suggestions and failed techniques are avoided. This knowledge
persists across sessions.

### Chain of Thought Patterns

Use these thinking patterns to reason about experiments:

**Bottleneck analysis:** "What is the weakest link? If I could magically
fix ONE thing, which change would give the biggest improvement?"

**Counterfactual reasoning:** "If I reverted experiment #N, would the
later experiments still work? If not, #N is a dependency."

**Diminishing returns detection:** "Each time I improve X, the gain gets
smaller. Is X approaching its optimal? Should I switch to Y?"

**Interaction hunting:** "A and B each improved the metric alone. What
happens if I combine them? If A+B > A + B, there's a positive interaction."

**Failure analysis:** "This change should have worked but didn't. What
assumption was wrong? Was the cause something I can observe in the data?"

### Research Memos (every 10 experiments)

Write to `.deepresearch/memos/memo-N.md`:

```markdown
# Research Memo — Experiments #N to #N+10

## Current Theory of the Metric
[1-2 sentences: what you believe drives the metric right now]

## Key Findings
[What the last 10 experiments revealed — specific, with numbers]

## Causal Model Update
[How your understanding of cause→effect changed. Draw the chain:]
[e.g., "LR schedule → convergence speed → final loss. Depth → capacity
→ but only useful if LR is already good (dependency discovered in exp #12)"]

## Dead Ends (stop trying)
[What to never try again, with evidence WHY it fails]

## Next Direction
[Where the next 10 experiments should focus. Be specific.]

## Open Questions
[What you DON'T understand yet. What experiment would answer it?]
```

These memos compound. Memo #3 references memo #1's theory and either
confirms or updates it. A session with good memos produces a coherent
research narrative. A session without them is just random search with a
prettier log.

### Causal Dependencies

Track which experiments depend on each other:
```json
{"experiment": 15, "depends_on": [7, 12],
 "reason": "Muon (exp 7) only works with arch 7 (exp 12). Reverting 12 would break 15."}
```
Use dependencies for: smart ablation (only test dependent changes),
smart crossover (don't combine conflicting dependencies), and session
handoff (next session knows which changes are load-bearing).

### When NOT to Think Deep

Not every experiment needs 5 minutes of reasoning. Scale thinking to
temperature and experiment phase:

- **T > 0.7 (exploring):** Full Deep Read + Hypothesis. You're searching
  for the right direction — thinking is high-value here.
- **T 0.3-0.7 (narrowing):** Brief hypothesis. You know the direction,
  just need to find the right magnitude.
- **T < 0.3 (fine-tuning):** Minimal hypothesis. "Nudging LR from 3e-4
  to 2.5e-4, expect <0.5% improvement." Don't overthink small changes.

---

## Level 2-3 — Generative Mutations (beyond parameter tuning)

Level 1 changes values in existing code. Level 2+ changes the CODE ITSELF.
This is the jump from optimizer to engineer. Benchmark proof: Level 3
outperforms Level 1 by **+189%** on identical problems (see benchmark_level3.py).

### Mutation Types

The agent has 6 mutation types, unlocked progressively:

| Type | Level | What it does | Example |
|---|---|---|---|
| `parametric` | 1 | Change a value | `DEPTH = 8 → 12` |
| `structural_addition` | 2 | Add new code block/function/class | Add caching layer, add retry logic |
| `structural_removal` | 2 | Remove dead code or unnecessary complexity | Simplify nested ifs to lookup table |
| `structural_replacement` | 2 | Replace one implementation with a better one | Linear search → hash map |
| `integration` | 2 | Connect two existing components that weren't connected | Wire cache into request handler |
| `architectural` | 3 | Design and implement a new component from spec | Build a plugin system from research |

**Safety rails for Level 2+ mutations:**
1. All existing tests MUST pass before AND after the mutation
2. New code must have at least one test
3. Revert immediately if any test breaks
4. Never modify read-only files (eval harness, test suite)
5. git commit before mutation, `git reset --hard` on failure

Configure in config.json:
```json
{
  "test_command": "pytest tests/ -q",
  "target_files": ["src/"],
  "read_only_files": ["tests/", "eval.sh"],
  "mutation_levels": [1, 2],
  "hard_constraints": ["all tests pass", "no new dependencies without approval"]
}
```

### Curriculum — Progressive Goals

Instead of one flat metric, define a sequence of goals in
`.deepresearch/curriculum.json`:

```json
{
  "stages": [
    {"name": "Correctness", "metric": "test_pass_rate", "target": 1.0, "direction": "higher"},
    {"name": "Performance", "metric": "p99_latency_ms", "target": 100, "direction": "lower"},
    {"name": "Scale", "metric": "max_concurrent", "target": 1000, "direction": "higher"}
  ]
}
```

The agent advances to the next stage only when the current target is met.
Each stage can focus on different mutation types: Stage 1 might use
structural additions (build the foundation), Stage 3 might use parametric
tuning (optimize what's already built).

Generate templates: `python engine/level3.py curriculum-init web_api`
Available: `web_api`, `ml_training`, `library`, `game`, `custom`.

### Domain Research Protocol (Level 2+ and Level 3)

Before coding or before a Level 2 structural mutation, the agent acquires
domain knowledge using the knowledge system:

```python
from engine.knowledge import KnowledgeAcquisition

ka = KnowledgeAcquisition(domain="web_api", spec="Optimize REST API", language="python")

# 1. Generate search queries (prioritized by bottleneck)
queries = ka.generate_searches(bottleneck="latency")

# 2. For each query: web_search → read top results → register + extract
for q in queries[:5]:
    # Agent uses web_search(q["query"]), reads results
    ka.register_source(url, title, source_type="documentation", relevance=0.9)
    ka.mark_source_read(url, summary="...", key_insights=["..."])
    ka.extract_technique(source_url=url, name="...", description="...",
                        expected_impact="...", evidence="...",
                        applicable_when="...", complexity="...")

# 3. Get prioritized technique list
next_technique = ka.suggest_next(current_bottleneck="DB latency")
context = ka.hypothesis_context("connection_pooling", "DB latency 142ms")

# 4. After each experiment, record the result
ka.record_result("connection_pooling", "worked: p99 from 142ms to 85ms")
```

**The 6-step research flow:**
1. **Understand the spec:** Input, output, constraints, metric
2. **Generate searches:** `ka.generate_searches()` — targeted queries by domain + bottleneck
3. **Read and extract:** 5-7 sources max, 2-3 min per source, extract techniques + pitfalls
4. **Build technique library:** The agent's domain-specific menu, built from research (not pre-defined)
5. **Define curriculum:** Progressive goals from correctness to optimization
6. **Begin experiments:** Use `ka.suggest_next()` and `ka.hypothesis_context()` in R2

All research persists in `.deepresearch/research/` (sources.json, techniques.json,
domain_knowledge.json). Next session continues where the last stopped.

### When to Use Which Level

- **Level 1** — The code works but could be faster/better. Knobs exist to turn.
- **Level 2** — The code is missing features. Adding capabilities would help more than tuning existing ones.
- **Level 3** — Starting from a specification. No code exists yet, or the existing code needs fundamental redesign.

The Reasoning Layer (R1 Deep Read) helps decide: if the bottleneck is
a parameter (LR too high, cache too small), use Level 1. If the bottleneck
is a missing capability (no caching at all, no error handling), use Level 2.

### Multi-File Scope (Level 2+)

Level 1 targets a single file. Level 2+ targets a **codebase**:

```json
{
  "target_files": ["src/"],
  "read_only_files": ["tests/", "eval.sh", "config/"],
  "entry_point": "src/main.py",
  "test_command": "pytest tests/ -q"
}
```

The agent reads ALL files in `target_files` during R1 (Deep Read),
but modifies only what's needed for each mutation. Multi-file mutations
(e.g., adding a new class in `utils.py` and importing it in `main.py`)
are a single experiment — atomic commit, atomic revert.

### Architecture Planning (Level 3)

Before coding from a specification, write `.deepresearch/architecture_plan.md`:

```markdown
## Components
1. [Name] — [purpose] — [estimated complexity]

## Dependency Order
1. [Foundation] (no deps) → build first
2. [Next] (depends on #1) → build second

## Key Design Decisions
- [Decision]: [option A] vs [option B], chose [X] because [reason]

## Test Strategy per Component
- [Component]: [how to verify it works in isolation]
```

The plan is itself an artifact that can be iterated. Experiment #1 might
be "implement the plan." Experiment #2 might be "the plan was wrong about
X, redesign the data flow." The Reasoning Layer treats the architecture
as a hypothesis to be tested, not a fixed blueprint.

Generate a plan template: `python engine/level3.py plan "your spec here"`

### Level 3 Thinking Protocol

At Level 3, the agent's thinking expands from "how to change this code"
to "how to design this system." The Reasoning Layer adapts:

**R1 at Level 3:** Read the specification, survey the domain, identify
the standard architecture, and understand WHY that architecture exists.
"Web servers use request/response + middleware because..."

**R2 at Level 3:** Hypothesize about architecture, not parameters.
"I think a producer-consumer pattern fits because the workload is
IO-bound with bursty writes. This connects to the domain research
which showed that similar systems use message queues."

**R3 at Level 3:** Reflect on architectural decisions, not just metrics.
"Adding the cache improved latency by 40%, but increased memory usage
by 200%. The architecture plan assumed memory was cheap — need to
revisit if we hit the memory constraint before the latency target."

**Memos at Level 3:** Track architectural evolution, not just parameter
tuning history. "Started with synchronous design. Experiment #15 showed
async is 3x faster. Redesigned data flow. Experiment #20 added connection
pooling on top of async — combined improvement: 5x. The synergy between
async and pooling wasn't in the original plan."

---

## Phase 1 — The Core Loop

This is the heartbeat. Every experiment follows this exact protocol.

**Unified decision flow (L1 through L3):**

```
1. CHECK CURRICULUM  → What stage am I in? What mutation strategy?
                       (python -m engine.level3 curriculum)
2. DEEP READ (R1)   → Understand the artifact. What's the bottleneck?
3. DECIDE LEVEL     → Is the bottleneck a parameter (L1) or a missing
                       capability (L2) or a missing component (L3)?
4. FORM HYPOTHESIS  → Theory + prediction + confidence
5. SELECT STRATEGY  → Bandit (python strategy.py select) tells WHICH category
                       Curriculum tells WHICH mutation type
6. MUTATE           → L1: change a value
                       L2: use MutationManager (from engine.mutations)
                       L3: use Orchestrator (from engine.autonomous)
7. EXECUTE + SCORE  → eval.sh (fixed budget) → metric
8. REFLECT (R3)     → Why? Update causal model
9. LOG              → experiments.jsonl + bandit update + knowledge base
```

The Level 1 strategy engine (strategy.py) and Level 2-3 engine (engine/)
work together: strategy.py decides WHICH category and HOW aggressively.
The engine decides WHAT TYPE of mutation and enforces safety rails.

### 1.1 Select Strategy (Bandit + Temperature)

Before each experiment, the Strategy Engine decides:
- **WHICH category** to mutate (bandit selection)
- **HOW aggressively** to mutate (temperature)
- **WHICH branch** to mutate from (population selection)

Use `python strategy.py select` to get the decision:
```bash
$ python strategy.py select
{"category": "architecture", "branch": "branch-1", "reason": "thompson_sampling (T=0.38)",
 "temperature": 0.38, "experiment_number": 7, "special_action": null}
# special_action can be: null, "crossover" (every 10), or "ablation" (every 20)
```

Use `python strategy.py status` for a dashboard:
```bash
$ python strategy.py status
=== DeepResearch Status ===
Total experiments: 42
Temperature: 0.1247
Baseline: 0.993
Best: 0.961
Improvement: +3.22%

--- Bandit Arms ---
  architecture         | trials= 14 | α=  8 β=  7 | success=50%
  hyperparameters      | trials= 10 | α=  4 β=  7 | success=30%
  optimizer            | trials=  8 | α=  2 β=  7 | success=13%
```

#### Bandit Selection — Thompson Sampling

Each mutation category is a "bandit arm" with a Beta distribution:
- α = number of successes (kept experiments) + 1
- β = number of failures (reverted experiments) + 1

To select the next category:
1. For each arm, sample from Beta(α, β)
2. Pick the arm with the highest sample
3. With probability = temperature, override and pick a RANDOM arm instead
   (forced exploration)

This naturally balances exploitation (categories that work) with exploration
(categories not yet tried enough). Early on, with high temperature, exploration
dominates. As temperature cools, the agent focuses on what works.

**Initialize arms** from the mutation_categories in config:
```json
{
  "architecture": {"alpha": 1, "beta": 1, "trials": 0},
  "hyperparameters": {"alpha": 1, "beta": 1, "trials": 0},
  "optimizer": {"alpha": 1, "beta": 1, "trials": 0}
}
```

#### Alternative: UCB1 (Upper Confidence Bound)

If you prefer a deterministic selection strategy over Thompson's stochastic
sampling, use UCB1. It selects the arm maximizing:

```
UCB1(arm) = (successes / trials) + C × sqrt(ln(total_trials) / trials)
```

where C controls exploration (default C=1.41, increase for more exploration).

**When to use which:**
- **Thompson Sampling** (default): Better when categories have very different
  success rates. Naturally adapts, less tuning needed. Preferred for most cases.
- **UCB1**: Better when you want deterministic, reproducible arm selection.
  Easier to debug. Good for benchmark comparisons where randomness is unwanted.

#### Temperature Schedule

Temperature controls the probability of random exploration and the acceptance
of worse results (simulated annealing):

```
T(n) = T_initial × (decay_rate ^ n)

"aggressive":    T_initial=1.0,  decay_rate=0.97  → T(50)≈0.22, T(100)≈0.05
"moderate":      T_initial=0.5,  decay_rate=0.95  → T(50)≈0.04, T(100)≈0.003
"conservative":  T_initial=0.2,  decay_rate=0.93  → T(50)≈0.005
```

At each experiment:
- Probability of forced random exploration = T(n) × 0.5
- Probability of accepting a WORSE result = exp(-Δ/T(n)) where Δ is the
  metric degradation normalized to baseline range

This means early experiments can accept small regressions to escape local
optima. Late experiments are nearly greedy — only improvements pass.

#### Population Selection

With K branches maintained, select which branch to mutate:

1. **Tournament selection**: Pick 2 random branches, mutate the better one
2. **With probability T(n)**: Pick a random branch instead (diversity)
3. **Every 10 experiments**: Try a CROSSOVER — combine changes from the top
   2 branches into a new variant

### 1.2 Form Hypothesis

Read:
- The selected branch's current state (the artifact files)
- The experiment log (what worked, what failed, what hasn't been tried)
- The knowledge base (cross-session patterns)

Then form a hypothesis: "Changing X in category Y will improve the metric
because Z." Write this to the experiment log BEFORE making the change.

**Hypothesis quality matters.** Don't just randomly tweak. Look for:
- Patterns in the log (every time we increased X, metric improved)
- Anti-patterns (reducing X always fails — stop trying)
- Unexplored combinations of successful individual changes
- Insights from the knowledge base (this worked in a similar domain)
- Near-misses (experiment #12 was only 0.1% worse — try a gentler variant)

### 1.3 Mutate

Make ONE focused change. Rules:
- Small and testable. Don't combine unrelated changes.
- The change must be reversible (git makes this trivial).
- Stay within the selected mutation category.
- Scale mutation type AND magnitude with temperature:

```
T > 0.7  (hot)   → Level 2 mutations. Add features, replace algorithms,
                    restructure code. Use MutationManager:
                    mm = MutationManager()
                    proposal = mm.propose("structural_addition", ["src/app.py"],
                        description="Add connection pooling",
                        hypothesis="DB connections are the bottleneck")
                    # Apply the change, then:
                    result = mm.execute(proposal)  # tests before/after, auto-revert

T 0.3–0.7 (warm) → Level 1-2 mix. Parameter changes + small structural additions.
                    Moderate scope: change a value OR add a small helper function.

T < 0.3  (cold)  → Level 1 only. Fine-tuning known-good parameters.
                    Small nudges. No structural changes.
```

**For Level 2+ mutations:** ALWAYS use the MutationManager safety rails.
It snapshots files before mutation, runs tests before AND after, and
auto-reverts if tests break. Direct file editing without safety rails
is only acceptable for Level 1 parametric changes.

### 1.4 Execute

Run the evaluation harness with fixed budget. Redirect ALL output.

```bash
# Run the eval harness (domain-agnostic — uses eval.sh you created in setup)
bash .deepresearch/eval.sh > .deepresearch/run.log 2>&1
EXIT_CODE=$?

# Extract metric (eval.sh must print "metric: <value>" as its last meaningful line)
METRIC=$(grep "^metric:" .deepresearch/run.log | tail -1 | awk '{print $2}')
```

If the run crashes (empty metric, non-zero exit):
- Read the last 50 lines of the log for the error
- If the error is obvious (typo, import, OOM), try ONE fix
- If the fix also crashes, revert and log as "crashed"
- Move on — don't waste more than 2 attempts on a crash

### 1.5 Score + Decide

Compare the new metric to the CURRENT BRANCH BEST (not just the global best):

**Case: Improved** (metric moved in the direction specified by `metric_direction`)
→ Keep the change. Git commit. Update branch best.
→ If this branch now beats the global best, update global best.
→ Update bandit arm: α += 1 (success for this category)
→ Save to knowledge base as a successful pattern

```bash
# Improvement check (works for both "lower" and "higher" directions)
DIRECTION=$(jq -r '.metric_direction' .deepresearch/config.json)
if [ "$DIRECTION" = "lower" ]; then
  IMPROVED=$(python3 -c "print('yes' if $NEW < $PREV_BEST else 'no')")
else
  IMPROVED=$(python3 -c "print('yes' if $NEW > $PREV_BEST else 'no')")
fi
```

**Case: Worse, but within annealing threshold**
→ Calculate acceptance probability: P = exp(-|Δ| / T(n))
→ Roll random [0,1]. If roll < P: ACCEPT the worse result anyway.
→ This is the simulated annealing escape hatch. Allows escaping local optima.
→ Log as "accepted-worse" with the probability that allowed it.
→ Bandit arm: β += 1 (still counts as a failure for arm statistics)

**Case: Worse, rejected**
→ Revert: `git reset --hard HEAD~1`
→ Update bandit arm: β += 1
→ Save to knowledge base as a failed approach (anti-pattern if repeated)

**Case: Crashed**
→ Revert. Log as crashed. Bandit arm: β += 1
→ Note the error type in the knowledge base

### 1.6 Log Everything

Append to `.deepresearch/experiments.jsonl`:

```json
{
  "id": 42,
  "timestamp": "2026-03-23T03:14:22+01:00",
  "session": "session-20260323",
  "branch": "branch-1",
  "category": "performance",
  "mutation_type": "structural_addition",
  "hypothesis": "Adding connection pooling should reduce p99 latency because each request currently creates a new DB connection (measured 40ms overhead per request)",
  "mutation_description": "Added ConnectionPool class to db.py, integrated into request handler",
  "metric": 85,
  "previous_best": 142,
  "improvement_pct": 40.1,
  "status": "kept",
  "temperature": 0.51,
  "acceptance_probability": null,
  "duration_seconds": 30,
  "reflection": "Confirmed DB connections were the bottleneck. Latency dropped from 142ms to 85ms. Next bottleneck is likely serialization — profile shows 30ms in JSON encoding.",
  "depends_on": []
}
```

**Required fields:** `id`, `timestamp`, `category`, `metric`, `status`, `branch`.
**Optional fields:** `hypothesis`, `mutation_description`, `previous_best`,
`improvement_pct`, `temperature`, `acceptance_probability`, `duration_seconds`,
`notes`, `session`.
**Valid status values:** `"baseline"`, `"kept"`, `"reverted"`, `"crashed"`, `"accepted-worse"`, `"skipped"`.

### 1.7 Periodic Checkpoints

**Every 5 experiments:** Write a brief status update to the log:
- Current best metric vs baseline (absolute and %)
- Bandit arm statistics (which categories are winning)
- Temperature status
- Population diversity (how different are the branches)

**Every 10 experiments:** Trigger a CROSSOVER attempt:

```bash
# 1. Get diffs of top 2 branches relative to baseline
git diff branch-0..branch-1 -- ${TARGET_FILES} > /tmp/diff-A.patch
git diff branch-0..branch-2 -- ${TARGET_FILES} > /tmp/diff-B.patch

# 2. Create crossover branch from baseline
git checkout -b deepresearch/crossover-${N} branch-0

# 3. Apply branch-A changes first (these are the better branch)
git apply /tmp/diff-A.patch || git apply --3way /tmp/diff-A.patch

# 4. Attempt to layer branch-B changes on top
git apply /tmp/diff-B.patch 2>/dev/null
# If conflicts: skip conflicting hunks, keep branch-A's version
# Use: git apply --reject /tmp/diff-B.patch && rm -f *.rej

# 5. Test the crossover as a normal experiment
bash .deepresearch/eval.sh > .deepresearch/run.log 2>&1
# If metric beats BOTH parents → new branch leader
# If not → delete crossover branch, revert
```

**Every 20 experiments:** Perform ABLATION on the best branch:

```bash
# 1. List all commits on best branch since baseline
COMMITS=$(git log --oneline branch-0..HEAD | awk '{print $1}')

# 2. For each commit, revert ONLY that commit and re-eval
for COMMIT in $COMMITS; do
  git stash
  git revert --no-commit $COMMIT
  bash .deepresearch/eval.sh > .deepresearch/run.log 2>&1
  ABLATED_METRIC=$(grep "^metric:" .deepresearch/run.log | tail -1 | awk '{print $2}')
  # If removing this commit HURTS the metric → commit is valuable, keep it
  # If removing this commit has NO EFFECT → commit is noise, mark for pruning
  git checkout -- .  # restore
  git stash pop
done

# 3. Interactive rebase to squash/drop noise commits
# git rebase -i branch-0  (drop commits marked as noise)
```

### 1.8 Population Management

Maintain exactly K branches. When a new variant is born (from crossover or
a particularly good random exploration), decide which branch to replace:

```bash
# Population replacement logic (called after a successful crossover or exploration)
python3 -c "
import json
state = json.load(open('.deepresearch/strategy-state.json'))
pop = state.get('population', [])
config = json.load(open('.deepresearch/config.json'))
direction = config.get('metric_direction', 'lower')
K = config.get('population_size', 3)

if len(pop) < K:
    print('SLOT_AVAILABLE: add new branch directly')
else:
    # Find global best — never replace it
    if direction == 'lower':
        best_idx = min(range(len(pop)), key=lambda i: pop[i]['metric'])
        worst_idx = max(range(len(pop)), key=lambda i: pop[i]['metric'])
    else:
        best_idx = max(range(len(pop)), key=lambda i: pop[i]['metric'])
        worst_idx = min(range(len(pop)), key=lambda i: pop[i]['metric'])

    # Check if all branches within 1%
    metrics = [p['metric'] for p in pop]
    spread = (max(metrics) - min(metrics)) / max(abs(min(metrics)), 1e-9)
    if spread < 0.01:
        # All similar — replace oldest
        oldest_idx = min(range(len(pop)), key=lambda i: pop[i].get('updated', 0))
        if oldest_idx != best_idx:
            print(f'REPLACE_OLDEST: branch {pop[oldest_idx][\"branch\"]}')
        else:
            print(f'REPLACE_WORST: branch {pop[worst_idx][\"branch\"]}')
    else:
        # Replace worst (never the best)
        if worst_idx != best_idx:
            print(f'REPLACE_WORST: branch {pop[worst_idx][\"branch\"]}')
"
```

---

### End-to-End Walkthrough — One Full Experiment (Level 2)

This shows experiment #7 on a web API optimization project. Demonstrates
the complete flow: curriculum → knowledge → R1 → R2 → strategy → mutate → eval → R3.

```
# ─── 0. CHECK CURRICULUM ───
python -m engine.level3 curriculum
# → Stage 2: Performance — p99_latency_ms: 142 / target 100 ← CURRENT
#   Strategy: types=['structural_addition','parametric'], T=0.5

# ─── 1. STRATEGY ENGINE ───
python strategy.py select
# → {"category": "performance", "branch": "branch-1", "temperature": 0.51}

# ─── 2. R1: DEEP READ ───
# Agent reads src/server.py, src/db.py, reviews last 5 experiments.
# "Request handling: 200ms total. Profiling shows 142ms in db.py:query().
# Each request opens a new PostgreSQL connection (TCP + TLS handshake).
# This is the bottleneck — not the application logic (58ms)."

# ─── 3. KNOWLEDGE CHECK ───
# Agent checks technique library:
python -m engine.level3 techniques
# → [0.80] connection_pooling: Reuse DB connections (untried)
#   Source: FastAPI Deployment Guide
#   Evidence: Benchmarks show 3x throughput
#   Applicable when: DB calls are the bottleneck ← MATCHES

# ─── 4. R2: HYPOTHESIS (knowledge-backed) ───
# "Bottleneck: DB connection overhead (142ms per request, from R1).
# Theory: Each request creates a new TCP+TLS connection. The FastAPI
# Deployment Guide (source #1) documents that connection pooling
# reduces this to near-zero. Benchmarks show 3x throughput.
# Experiment: Add sqlalchemy connection pool (pool_size=20).
# Prediction: p99 drops from 142ms to ~60-80ms.
# Connects to: exp #3 confirmed DB is the bottleneck (profiling).
# Risk: pool exhaustion under load → add max_overflow=10.
# Confidence: high — profiling data + external benchmarks agree."

# ─── 5. MUTATE (Level 2 structural_addition) ───
# Agent uses MutationManager for safety:
#   proposal = mm.propose("structural_addition", ["src/db.py"],
#       description="Add connection pool with pool_size=20",
#       hypothesis="DB connections are bottleneck, pooling reduces to near-zero")
#   # Agent writes the code (adds ConnectionPool class, integrates into query())
#   result = mm.execute(proposal)  # pre-test → apply → post-test → auto-revert if broken

# ─── 6. EXECUTE EVAL ───
bash .deepresearch/eval.sh
# → metric: 85  (was: 142)

# ─── 7. SCORE + DECIDE ───
# 85 < 142 (lower is better) → KEPT (+40.1%)
git add src/db.py && git commit -m "deepresearch #7: structural_addition — connection pooling (p99=85ms)"

# ─── 8. R3: REFLECTION ───
# "Predicted 60-80ms, got 85ms. Theory confirmed but slightly
# underperformed — likely because pool_size=20 is suboptimal for our
# load pattern. Bottleneck SHIFTED: DB is now 85ms but serialization
# is 58ms (from profiling). Next: try orjson for faster JSON encoding
# (parametric change, T=0.51 supports moderate change).
# This experiment DEPENDS ON nothing — connection pooling is independent."
ka.record_result("connection_pooling", "worked: p99 from 142ms to 85ms")

# ─── 9. LOG ───
python strategy.py update '{"id":7,"category":"performance","branch":"branch-1",
  "mutation_type":"structural_addition","metric":85,"previous_best":142,
  "improvement_pct":40.1,"status":"kept","hypothesis":"Connection pooling...",
  "reflection":"Theory confirmed, bottleneck shifted to serialization",
  "depends_on":[]}'

# ─── 10. CURRICULUM CHECK ───
python -m engine.level3 curriculum
# → Stage 2: Performance — p99_latency_ms: 85 / target 100 ✅ COMPLETE!
# → Advanced to Stage 3: Load handling — max_concurrent: ? / 1000

# ─── LOOP → experiment #8 starts at step 0
```

---

## Phase 2 — Strategy Engine Details

### Intelligent Exploration Strategies

Beyond the basic bandit + annealing, the agent should watch for these patterns
and act on them using the Reasoning Layer:

**Momentum tracking:** If the last 3 experiments in a category all improved,
go bigger in that category. If 3 consecutive failures, skip it for now.
Check: read last 15 experiments from the log, group by category.

**Plateau detection:** If 5+ consecutive experiments show <0.1% change, the
agent is in a flat region. Action: reheat temperature to `min(T * 3, 0.8)`,
try an unused category, attempt a Level 2 structural mutation.

**Regression analysis (every 20 experiments):** Use `python strategy.py status`
and read the experiment log to answer:
1. Which categories have the best success rate?
2. Which pairs of categories improve when done sequentially? (interaction effects)
3. Where do the biggest gains come from?

The Reasoning Layer (R1) should INTERPRET these patterns — don't just follow
numbers. "Architecture has 50% success rate" is less useful than "architecture
improvements work because the model was underfitting, and now it's not."

**Guided random restarts (if stuck 15+ experiments):**
1. Identify top-3 most impactful experiments from the log
2. Reset to baseline: `git checkout deepresearch/branch-0 -- $TARGET_FILES`
3. Cherry-pick only those 3 changes (keeps proven wins, discards noise)
4. Reheat temperature to 0.8, continue from this cleaner base

---

## Phase 3 — Persistent Memory

### Knowledge Base Schema

`.deepresearch/knowledge.json` accumulates insights across sessions:

```json
{
  "patterns": [
    {
      "domain": "code_optimization",
      "category": "performance",
      "description": "Adding connection pooling consistently reduces p99 latency by 30-50% when DB calls are the bottleneck",
      "confidence": 0.9,
      "evidence_count": 5,
      "first_seen": "2026-03-20",
      "last_confirmed": "2026-03-23"
    },
    {
      "domain": "prompt_optimization",
      "category": "examples",
      "description": "Adding 3 few-shot examples consistently improves judge score by 1-2 points",
      "confidence": 0.9,
      "evidence_count": 5,
      "first_seen": "2026-03-18",
      "last_confirmed": "2026-03-23"
    }
  ],
  "anti_patterns": [
    {
      "domain": "code_optimization",
      "category": "parallelism",
      "description": "Adding threads to I/O-bound Python functions with GIL makes it slower, not faster. Use async instead.",
      "confidence": 0.95,
      "evidence_count": 4
    },
    {
      "domain": "game_balancing",
      "category": "economy",
      "description": "Resource generation rates above 1.5x baseline always cause hyperinflation by turn 50",
      "confidence": 0.8,
      "evidence_count": 4
    }
  ],
  "domain_insights": [
    {
      "domain": "web_api",
      "insight": "For APIs with mixed read/write load, caching + connection pooling together give 3x more improvement than either alone (synergy)",
      "source_session": "session-20260322",
      "metric_impact": "p99 from 300ms to 45ms"
    }
  ],
  "cross_domain": [
    {
      "pattern": "Reducing complexity before adding it back yields better results than iterative addition alone",
      "domains_seen": ["ml_training", "prompt_optimization"],
      "confidence": 0.7
    }
  ]
}
```

### Memory Update Protocol

`python strategy.py update <result_json>` handles all knowledge updates automatically:
- **Pattern detection:** 3+ successes in a category → recorded as pattern
- **Anti-pattern detection:** 3 consecutive failures → recorded as anti-pattern (stop trying)
- **Domain insights:** at session end, best category and key findings are stored

At session START, `strategy.py select` automatically loads knowledge.json,
biases bandit priors from past patterns, skips anti-pattern categories,
and prints relevant insights for context.

The Knowledge Acquisition system (`engine/knowledge.py`) adds a second layer:
techniques extracted from external research are stored in `.deepresearch/research/techniques.json`
and their results (worked/failed) persist across sessions. This prevents the agent from
re-trying techniques that already failed and reinforces techniques that worked.

### Cross-Session Continuity

When starting a new session in the same project:
1. Load `.deepresearch/strategy-state.json` — resume temperature, bandit stats
2. Load `.deepresearch/experiments.jsonl` — full history
3. Load population snapshots — continue from the best branches
4. Increment session_count in config.json

The agent should read the last session's report (if any) and use it as
context for forming the first hypothesis of the new session.

---

## Phase 4 — Parallel Experiments

When eval is fast (<60s), run multiple experiments concurrently using git
worktrees. This works for code optimization, prompt engineering, and game
balancing. For GPU-bound ML training, use CUDA_VISIBLE_DEVICES instead.

### Method A — Git Worktrees (code, prompts, configs)

```bash
# 1. Create parallel worktrees from different branches
K=3  # population size
for i in $(seq 0 $((K-1))); do
  git worktree add /tmp/dr-branch-${i} deepresearch/branch-${i} 2>/dev/null || true
done

# 2. Apply different mutations to each worktree
# (Agent decides mutation per branch using strategy.py select)

# 3. Run evals in parallel
for i in $(seq 0 $((K-1))); do
  (
    cd /tmp/dr-branch-${i}
    cp ${PROJECT_ROOT}/.deepresearch/eval.sh .
    bash eval.sh > /tmp/dr-result-${i}.log 2>&1
    METRIC=$(grep "^metric:" /tmp/dr-result-${i}.log | tail -1 | awk '{print $2}')
    echo "{\"branch\":\"branch-${i}\",\"metric\":${METRIC}}" > /tmp/dr-result-${i}.json
  ) &
done
wait  # All experiments finish

# 4. Collect results and update strategy state
for i in $(seq 0 $((K-1))); do
  RESULT=$(cat /tmp/dr-result-${i}.json)
  python strategy.py update "$RESULT"
done

# 5. Cleanup worktrees when done
for i in $(seq 0 $((K-1))); do
  git worktree remove /tmp/dr-branch-${i} --force 2>/dev/null || true
done
```

### Method B — Multi-GPU (ML training)

```bash
# Each experiment on a different GPU
for GPU in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$GPU
    cd /tmp/dr-gpu-${GPU}
    bash .deepresearch/eval.sh > /tmp/dr-gpu-${GPU}-result.log 2>&1
  ) &
done
wait
```

### Method C — Sequential with Population (default fallback)

If parallel execution isn't available, the agent runs experiments sequentially
but still maintains K branches and rotates between them using tournament
selection. The search strategy is identical — just slower. This is the default
and requires no special setup.

### Speedup Expectations

| Parallel workers | Experiments/hour (60s eval) | vs sequential |
|---|---|---|
| 1 (sequential) | ~55 | 1× |
| 3 (worktrees) | ~150 | 2.7× |
| 5 (worktrees) | ~220 | 4× |
| 4 GPUs (ML, 5min) | ~48 | 4× |

---

## Phase 5 — Research Reports

### Auto-Generation Triggers

Generate a report when:
- A session ends (user interrupt, target reached, or context limit)
- Every 25 experiments within a session (progress report)
- A significant breakthrough occurs (>5% improvement in one experiment)

To generate: `python strategy.py report` — this reads experiments.jsonl and
strategy-state.json and writes the full report to `.deepresearch/reports/`.

The report includes: executive summary, baseline→best metric, top 10 changes,
failed approaches, bandit arm performance, and recommendations for next session.
The agent can also write custom sections based on its research memos.

---

## Domain Configurations

Initialize with `bash init.sh --domain <name>` — sets target, metric, categories, and curriculum automatically.

| Domain | Target | Metric | Direction | Key mutation categories |
|---|---|---|---|---|
| `ml` | train.py | val_loss | lower | architecture, hyperparameters, optimizer, regularization, scheduling, efficiency |
| `web_api` | src/ | p99_latency_ms | lower | algorithm, caching, connection_pooling, async, error_handling |
| `code` | target.py | benchmark_time | lower | algorithm, memory, parallelism, io, architecture |
| `prompt` | prompt.txt | judge_score | higher | structure, specificity, tone, examples, guardrails, persona |
| `game` | src/ | composite | higher | economy, combat, progression, map_balance, ai_behavior |
| `library` | src/ | benchmark_ops_sec | higher | algorithm, api_design, error_handling, data_structures |
| `custom` | src/ | primary_metric | higher | (you define) |

Each domain also gets a curriculum template (see `python -m engine.level3 curriculum <domain>`).

For Level 2+, use `python -m engine.level3 knowledge --domain <name>` to get domain-specific search queries for external knowledge acquisition.

---

### Quick Start Example — Prompt Optimization

End-to-end setup for optimizing a system prompt using LLM-as-judge:

```bash
# 1. Init
bash init.sh --domain prompt
# Edit config: target_files=["system_prompt.md"], metric=judge_score, direction=higher

# 2. Create eval harness with 5 test cases
cat > .deepresearch/eval.sh << 'EVAL'
#!/bin/bash
set -e
PROMPT=$(cat system_prompt.md)
TOTAL=0
for TEST in "Summarize this article" "Write a haiku" "Explain quantum computing" "Debug this code" "Translate to French"; do
  SCORE=$(curl -s https://api.anthropic.com/v1/messages \
    -H "Content-Type: application/json" -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d "$(jq -n --arg p "$PROMPT" --arg t "$TEST" '{
      model:"claude-sonnet-4-6", max_tokens:500,
      system: $p,
      messages:[{role:"user",content:$t}]
    }')" | jq -r '.content[0].text' | head -c 2000 | \
    curl -s https://api.anthropic.com/v1/messages \
    -H "Content-Type: application/json" -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d "$(jq -n --arg r "$(cat)" '{
      model:"claude-sonnet-4-6", max_tokens:50,
      messages:[{role:"user",content:("Rate 0-10, respond ONLY with the number:\n\n" + $r)}]
    }')" | jq -r '.content[0].text' | grep -oP '^\d+')
  TOTAL=$((TOTAL + ${SCORE:-0}))
done
echo "metric: $((TOTAL / 5))"
EVAL
chmod +x .deepresearch/eval.sh

# 3. Run baseline
bash .deepresearch/eval.sh  # → metric: 6

# 4. Start the loop
# Agent: "Read SKILL.md, start deepresearch on system_prompt.md"
```

### Quick Start Example — Level 2 (Optimize an existing codebase)

Use Level 2 when the code WORKS but is missing features that would improve it:

```bash
# 1. Init with curriculum
python -m engine.level3 init --spec "Optimize API performance" --domain web_api

# 2. Configure target files
# Edit .deepresearch/config.json:
#   target_files: ["src/"], test_command: "pytest tests/ -q",
#   metric: "p99_latency_ms", metric_direction: "lower",
#   mutation_levels: [1, 2]

# 3. Run baseline
bash .deepresearch/eval.sh  # → metric: 450 (ms)

# 4. Discover improvement opportunities
python -m engine.level3 discover
# → Agent analyzes codebase for: caching, batching, connection pooling,
#   async IO, error handling, input validation, dead code...

# 5. Start the loop with Level 2 mutations enabled
# Agent: "Read SKILL.md. Run deepresearch on src/. Use Level 2 mutations —
#   you can ADD code (caching, pooling, etc.) not just tune parameters.
#   Follow the curriculum stages: correctness first, then performance."

# 6. Check curriculum progress
python -m engine.level3 curriculum
# → Stage 1: Correctness ✅ | Stage 2: Performance 🔶 (450ms → target <100ms)
```

### Quick Start Example — Level 3 (Build from specification)

Use Level 3 when starting from a spec or when fundamental redesign is needed:

```bash
# 1. Init with spec
python -m engine.level3 init \
  --spec "Build a CLI tool that converts CSV to JSON with streaming, validation, and error recovery" \
  --domain library

# 2. Research phase (agent reads the spec, surveys existing tools)
python -m engine.level3 next
# → "Phase: research. Analyze the specification and answer: ..."
# Agent researches, then: python -m engine.level3 research  (to see progress)

# 3. Architecture phase (agent designs components)
python -m engine.level3 next
# → "Phase: architect. Design components with dependencies..."
# Agent designs: Parser, Validator, Converter, StreamWriter, ErrorHandler

# 4. Bootstrap (creates project structure with stubs)
python -m engine.level3 bootstrap
# → Creates src/*.py stubs, tests/*.py stubs, config files

# 5. Build phase (agent implements each component using DeepResearch loop)
python -m engine.level3 next
# → "Phase: build. Implement 'Parser': parse CSV rows with streaming..."
# Agent writes code, runs tests, iterates. Repeat for each component.

# 6. Optimize phase (curriculum-driven optimization)
python -m engine.level3 next
# → "Phase: optimize. Current stage: Performance — target 10K rows/sec"

# 7. Full status at any time
python -m engine.level3 status
```

---

## Stopping Conditions

The loop runs until ONE of these:

1. **User interrupts** — human says stop or Ctrl+C
2. **Target reached** — if the human defined a target metric value
3. **Plateau detected** — 15+ consecutive experiments with no improvement
   AND temperature < 0.05 AND at least 30 total experiments. Before stopping,
   try ONE guided random restart. If that also plateaus in 5 more experiments,
   stop and report.
4. **Context limit** — approaching context window limits. Write a comprehensive
   handoff report and stop gracefully. The next session can resume.
5. **Time limit** — if the human specified a max session duration

### Context Window Management

AI agents have finite context. Protect it aggressively:

- **Never read full logs.** Use `grep`, `tail -5`, `head -20` — never `cat`.
- **Redirect all experiment output** to `.deepresearch/run.log`. Extract only
  the metric line. Read error details only on crash.
- **Summarize, don't accumulate.** After every 10 experiments, write a 5-line
  summary of findings to `.deepresearch/session-notes.md` and stop carrying
  the raw experiment details in your working memory.
- **Load selectively.** Don't read the entire `experiments.jsonl` — use
  `tail -10` for recent history and `python strategy.py status` for stats.
- **Handoff early.** If you estimate you've used >70% of context, trigger the
  handoff report NOW rather than risking a mid-experiment cutoff. The report
  contains everything the next session needs to continue seamlessly.

When stopping for ANY reason:
1. Generate the full research report
2. Save strategy state (so next session can resume)
3. Update knowledge base with session findings
4. Commit the best state on the main branch
5. Present the report to the human

---

## Error Recovery

At session start, validate state files: config.json (required keys), strategy-state.json
(temperature, arms, population), experiments.jsonl (valid JSON per line), knowledge.json
(arrays exist). If corrupted:

1. `config.json` → re-run `init.sh` or restore from git
2. `strategy-state.json` → rebuild from experiments.jsonl: `python strategy.py status` shows what's recoverable. The experiment log is the source of truth — bandit arms and temperature can be recomputed from it.
3. `experiments.jsonl` bad lines → `python3 -c "import json; lines=open('.deepresearch/experiments.jsonl').readlines(); open('.deepresearch/experiments.jsonl','w').writelines(l for l in lines if l.strip() and json.loads(l))"`
4. `knowledge.json` → reset: `echo '{"patterns":[],"anti_patterns":[],"domain_insights":[],"cross_domain":[]}' > .deepresearch/knowledge.json`

**Backups:** Every 10 experiments: `mkdir -p .deepresearch/backups/$(date +%s) && cp .deepresearch/*.json .deepresearch/backups/$(date +%s)/`

**Crash recovery:** experiments.jsonl is append-only (no data loss), strategy-state.json
saved after every experiment, population snapshots are git-managed. Next session loads
all state and continues automatically.

**Git safety:** `git stash` before experiments, `git reset --hard` on revert, `.deepresearch/` in `.gitignore`, branches named `deepresearch/session-YYYYMMDD-HHMM`.

**Rate limiting:** Track API usage in strategy-state.json, exponential backoff on limits.

### Safety Guardrails

**Eval harness integrity:** The agent MUST NOT modify any file used by the
evaluation harness. Before each experiment, verify the eval harness checksum:
```bash
# At session start, record eval harness hash
sha256sum .deepresearch/eval.sh > .deepresearch/.eval-hash
# Before each experiment, verify
sha256sum -c .deepresearch/.eval-hash --quiet || { echo "ABORT: eval harness modified"; exit 1; }
```

**Runaway process protection:** Always wrap execution with `timeout`:
```bash
timeout $((BUDGET + 60))s bash .deepresearch/eval.sh "$BUDGET" "$TARGET" > .deepresearch/run.log 2>&1
# Extra 60s buffer for startup/shutdown. If it exceeds this, kill it.
```

**Resource limits:** For ML experiments, cap VRAM and prevent OOM cascades:
```bash
# Set max GPU memory (if applicable)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Monitor: abort if disk fills up
DISK_FREE=$(df -BM --output=avail . | tail -1 | tr -d ' M')
[ "$DISK_FREE" -lt 500 ] && { echo "ABORT: <500MB disk free"; exit 1; }
```

**Token/secret safety:** If the eval harness uses API keys (LLM-as-judge):
- Keys MUST come from environment variables, never hardcoded in any file
- Never log API keys — grep and redact before writing to experiment logs
- The `.deepresearch/` directory is `.gitignore`'d, but double-check before any push

**Mutation scope enforcement:** The agent may ONLY modify files listed in
`config.json → target_files`. Any attempt to edit other files is a bug.
Before each git commit, verify: `git diff --name-only` shows only target files.

**Goodhart's Law detection:** When a metric improves suspiciously fast (>10%
in one experiment), or the metric improves but the artifact quality clearly
degrades, the agent may be gaming the eval rather than genuinely improving.
Signs of metric gaming:
- LLM-as-judge: artifact gets shorter/simpler but score rises (judge is lenient on short outputs)
- Code benchmark: function gets faster by returning wrong results
- Test suite: tests pass by catching exceptions instead of fixing bugs

Mitigation: every 25 experiments, run a **sanity check** — have the agent
read the current best artifact and confirm it still makes sense. If it
looks degenerate, pause and flag for human review. The eval harness may need
refinement (but never change it mid-session — start a new session with a
better harness).

---

## Core Principles

```
Level 1 (always):
  ✓ Mutate from BRANCH BEST, not from last attempt
  ✓ ONE change per experiment
  ✓ NEVER modify the evaluation harness
  ✓ NEVER ask "should I continue?" — run autonomously
  ✓ Log EVERYTHING — negative results are data
  ✓ Use the knowledge base — don't repeat known failures
  ✓ Temperature controls boldness — be bold early, precise late

Level 2-3 (additional):
  ✓ THINK before mutating (R1 → R2 → R3 every experiment)
  ✓ Tests before AND after every structural mutation
  ✓ Auto-revert if tests break — no exceptions
  ✓ Curriculum stages are sequential — no skipping
  ✓ Never regress a completed curriculum stage
  ✓ Add tests WITH features, not after
  ✓ Research before building (Level 3)
  ✓ Build in dependency order — foundations first
```

---

## Edge Cases

**Resuming with zero experiments:** If `.deepresearch/experiments.jsonl` is
empty but config exists, re-run baseline (Phase 0.3) before entering the loop.
Don't assume the baseline exists — verify `baseline_metric` is set in state.

**Single target file vs multiple:** If `target_files` has one entry, standard
git diff/commit works. For multiple files, ensure ALL target files are staged
together — partial commits create inconsistent states. Use:
```bash
git add ${TARGET_FILES[@]} && git commit -m "..."
```

**Conflicting branches after crossover:** If merging two branches produces
git conflicts, do NOT attempt auto-resolution. Instead: apply each branch's
changes manually by reading both diffs, combining compatible changes, and
skipping incompatible ones. Test the result as a new experiment.

**Population size 1:** Degrades gracefully to standard autoresearch behavior.
No crossover, no tournament selection. Bandit + annealing still work.

**Metric returns NaN/Inf:** Treat as a crash. Revert and log:
```bash
if echo "$METRIC" | grep -qE 'nan|inf|NaN|Inf'; then
  echo "metric: CRASHED (NaN/Inf)"
  # revert
fi
```

**Eval harness slower than budget:** If eval takes longer than `budget_seconds`
(startup overhead, large models), the `timeout` wrapper kills it. Log as crashed
with note "timeout exceeded." If 3+ consecutive timeouts, the agent should
reduce model/data size in its next mutation.

**Level 2 mutation breaks tests that were passing:** This is expected — the
MutationManager auto-reverts. But if it keeps happening (3+ reverts in a row
on the same feature addition), the agent should: (a) read the failing test to
understand WHY, (b) try a smaller version of the same feature, or (c) add the
feature's test FIRST, then implement to make it pass (TDD approach).

**Level 3 research phase takes too long:** The agent should spend at most
5-10 minutes on domain research. If it spirals into reading too many sources,
set a hard limit: "Research 3 sources max, then move to architecture."

**Curriculum stage regression:** If optimizing Stage 2 (performance) breaks
Stage 1 (correctness), the curriculum runner flags it. The agent MUST fix
the regression before continuing. Use: `python -m engine.level3 curriculum`
to check for regressions.

**Level 2 mutation creates circular dependency:** If adding Feature B depends
on Feature A which depends on Feature B, don't try to add both at once.
Instead: add Feature A with a stub interface for B, test, then add B.

### Troubleshooting — Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| First 10 experiments all reverted | Baseline is already near-optimal OR mutations too aggressive | Switch to `"conservative"` temperature, try finer-grained categories |
| Metric oscillates ±0.1% forever | Measurement noise exceeds mutation impact | Increase eval budget (more test cases, longer runs) to reduce variance |
| Every crossover crashes | Branches diverged structurally (incompatible changes) | Reset one branch to baseline + top-3, reduce divergence |
| Strategy always picks same category | One arm has high α from early luck, others starved | Manually reheat: set temperature to 0.8 in strategy-state.json |
| Metric is CRASHED but no error in log | Eval script doesn't print "metric:" line | Ensure eval.sh always ends with `echo "metric: $VALUE"` |
| L2 mutation always breaks tests | Feature too complex for one mutation | Break into smaller sub-features, add tests for each separately |
| Curriculum stuck on stage 1 | Tests are too strict or metric target unrealistic | Review targets. Lower the bar for early stages to unblock progress |
| L3 architect phase produces too many components | Agent over-engineers | Cap at 5-7 components for v1. Add more in optimization phase |
| `python -m engine.level3 next` shows wrong phase | Orchestrator state stale | Check `.deepresearch/orchestrator_state.json`, advance manually if stuck |

---

## Scaling to 100+ Experiments

**Log rotation:** `experiments.jsonl` grows ~500 bytes per experiment. At 1000+
experiments, reading the full log each iteration slows context loading. Solution:
keep a rolling summary of the last 50 experiments in memory, use `tail -50` on
the full log, and consult the full log only for regression analysis (every 20
experiments).

**Knowledge base pruning:** After 500+ experiments, patterns and anti-patterns
accumulate. Prune entries with `confidence < 0.3` or `evidence_count < 2`.

**Branch cleanup:** Dead branches (replaced in population management) should be
deleted after 50 experiments: `git branch -D deepresearch/dead-branch-name`.
Keep only active population branches + main.

**Session handoff for multi-day runs:** If running across multiple days/sessions,
each session's report serves as context for the next. The agent should read the
most recent report in `.deepresearch/reports/` at session start and use its
"Recommendations" section to seed the first hypothesis.

**Diminishing returns curve:** Track cumulative improvement per experiment.
If the last 30 experiments averaged <0.05% improvement per kept experiment,
the optimization is near convergence. Report this and suggest either:
(a) a different metric, (b) a larger search space, or (c) declaring victory.

---

## Self-Improvement

This skill can be optimized using its own loop — target this SKILL.md and
engine/*.py, score via composite quality across 3+ test optimization tasks
from different domains. The meta-loop: the research engine improving its
own methodology.

### Self-Improvement Config

```json
{
  "target_files": ["SKILL.md", "engine/"],
  "metric": "composite_quality",
  "metric_direction": "higher",
  "budget_seconds": 300,
  "population_size": 1,
  "temperature_schedule": "conservative",
  "mutation_categories": ["clarity", "completeness", "correctness", "conciseness", "actionability"]
}
```

### Eval Criteria (LLM-as-judge rubric)

Score SKILL.md on these 5 criteria (0=fail, 1=pass each, total /5 × 10):

1. **CLEAR** — Instructions are unambiguous. An agent can follow them without guessing.
2. **COMPLETE** — All necessary steps, edge cases, and protocols are covered.
3. **CORRECT** — No bugs in code templates, no stale references, no contradictions.
4. **CONCISE** — No redundancy. Every section earns its place. No bloat.
5. **ACTIONABLE** — Contains copy-paste-ready commands, configs, and code.

Test across 3+ domains (e.g., ML training, web API, prompt engineering) to
ensure changes don't overfit to one domain's perspective.

### Safety Rails for Meta-Optimization

- **Never delete the Self-Improvement section** — the skill must remain self-improvable.
- **Preserve the 7-layer architecture** — individual layers can be improved, not removed.
- **Keep all eval templates functional** — test bash syntax after any template change.
- **Run `python test_integration.py`** after any engine/*.py change — all tests must pass.
- **Don't collapse good/bad examples** — they're high-value teaching material.

For Level 2-3 self-improvement: use the FeatureDiscovery patterns to
analyze the engine code itself. Are there missing error handlers? Could
the curriculum system be more flexible? Is the mutation manager missing
a mutation type? Apply Level 2 structural mutations to the engine.

---

## Implementation Notes for Claude Code

### Starting a Session

When the human says "run deepresearch" or similar:

1. Check if `.deepresearch/` exists
   - YES: Load state, check `python -m engine.level3 status`, show progress, ask "Resume or fresh start?"
   - NO: Determine the level needed:
     - Human has existing code + metric → Level 1-2 (run `init.sh`)
     - Human has a specification → Level 3 (run `python -m engine.level3 init`)
2. After confirmation, begin the core loop — **never ask again**
3. Print a one-line status after each experiment:
   `[#42 | branch-1 | structural_addition | T=0.51] metric: 142ms → 85ms ✓ kept (+40.1%)`
4. For Level 2+ mutations, always use the MutationManager safety rails
5. Check curriculum progress every 10 experiments: `python -m engine.level3 curriculum`
6. Use `grep`, `tail`, `head` — never `cat` on large logs
