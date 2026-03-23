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

Five layers:

1. **Reasoning Layer** — Deep read → causal hypothesis → reflection (NEW — the core differentiator)
2. **Strategy Engine** — Bayesian bandit + simulated annealing as researcher tools
3. **Population Search** — Multiple competing branches with smart crossover
4. **Persistent Memory** — Knowledge, patterns, anti-patterns, causal dependencies
5. **Research Reports** — Auto-generated memos and session reports

**Navigation** (for agents parsing this file):
- **Phase 0** — Setup (config, eval harness templates, baseline)
- **Reasoning Layer** — Deep Read, Causal Hypothesis, Reflection, Memos (READ THIS FIRST)
- **Phase 1** — Core Loop (select → hypothesize → mutate → execute → score → log)
- **Phase 1 Walkthrough** — End-to-end example with exact commands
- **Phase 2** — Strategy Engine (momentum, plateau detection, regression, restarts)
- **Phase 3** — Persistent Memory (knowledge.json schema, update protocol)
- **Phase 4** — Parallel Experiments (git worktrees, multi-GPU)
- **Phase 5** — Research Reports (templates, auto-generation triggers)
- **Domain Configurations** — ML, Code, Prompt, Game, Document presets
- **Stopping Conditions** — When and how to end
- **Error Recovery** — Validation, corruption repair, backups, safety

The human's job: define what "better" means, set constraints, write `research.md`.
The agent's job: everything else.

---

## Directory Structure

DeepResearch uses a `.deepresearch/` directory at the project root for all
persistent state. This directory is THE brain — it persists across sessions.

```
.deepresearch/
├── config.json          # Session config (metric, target, budget, etc.)
├── knowledge.json       # Cross-session knowledge base
├── dependencies.json    # Causal dependency graph between experiments
├── experiments.jsonl     # Append-only experiment log (one JSON per line)
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
autoresearch's train.py) or a small set of related files. Fewer is better.

**Primary metric:** ONE number. Direction must be clear (lower/higher = better).
Examples: val_bpb (lower), test pass rate (higher), response latency ms (lower),
LLM judge score 0-10 (higher), composite weighted score (higher).

**Evaluation harness:** How to compute the metric. This is IMMUTABLE during the
loop. Can be: a script, a test suite, an LLM-as-judge prompt, a benchmark.
Write it to `.deepresearch/eval.sh` or `.deepresearch/eval.py`.

Use one of these ready-made templates:

**Template A — Script metric (ML training, code benchmarks):**
```bash
#!/bin/bash
# .deepresearch/eval.sh — extract a single number from a script run
set -e
BUDGET="${1:-300}"
TARGET="${2:-train.py}"
LOG=".deepresearch/run.log"
timeout "${BUDGET}s" uv run "$TARGET" > "$LOG" 2>&1 || true
METRIC=$(grep "^val_bpb:\|^score:\|^result:" "$LOG" | tail -1 | awk '{print $2}')
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
    model: "claude-sonnet-4-20250514",
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

# Run eval (example for ML training)
timeout ${BUDGET}s uv run train.py > .deepresearch/run.log 2>&1

# Extract metric
BASELINE=$(grep "^val_bpb:" .deepresearch/run.log | awk '{print $2}')

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
Reasoning Layer wraps three thinking steps around every experiment:

### R1: Deep Read (before mutating)

Don't grep for numbers. UNDERSTAND the artifact:
1. Read the target file completely. What does each section do?
2. Identify the **bottleneck** — what limits the metric RIGHT NOW?
3. Check: has this bottleneck been addressed? Why didn't it work?
4. Form a **causal model**: "The metric is X because of Y."

### R2: Causal Hypothesis (replaces "pick a random category")

The bandit INFORMS but doesn't DICTATE. Write before making changes:

```
## Hypothesis #N
**Theory:** Metric limited by [bottleneck] because [cause].
**Prediction:** Changing [thing] improves metric ~[amount] because [mechanism].
**Connects to:** Builds on experiment #X which showed [finding].
**Risk:** Fails if [condition]. Fallback: [alternative].
**Confidence:** [low/medium/high]
```

The shift: autoresearch says "bandit picked architecture, change something."
DeepResearch says "I read the code, the bottleneck is quadratic attention,
experiments #5 and #8 showed efficiency changes compound, so I should try
windowed attention. This is an architecture change but the REASON matters."

### R3: Reflection (after seeing results)

Don't just update a counter. THINK:
1. Was my prediction correct? Direction? Magnitude?
2. If YES → what's confirmed? What's the next logical experiment?
3. If NO → was my theory wrong, implementation off, or interaction effect?
4. What did I LEARN? Has the bottleneck shifted?

Write a 2-3 sentence reflection. "Experiment #12 confirmed depth scaling
has diminishing returns past 12 layers. Bottleneck shifted from capacity
to optimization — next target: LR schedule." NOT just "Metric improved. Kept."

### Research Memos (every 10 experiments)

Write to `.deepresearch/memos/memo-N.md`:
- **Current theory** — what you believe drives the metric
- **Key findings** — what the last 10 experiments revealed
- **Causal updates** — how your understanding changed
- **Dead ends** — what to stop trying, and WHY
- **Next direction** — where to focus next

These memos are the most valuable output. They compound — memo #5
references #3 which references #1, creating a coherent research narrative
across the entire session.

### Causal Dependencies

Track which experiments depend on each other:
```json
{"experiment": 15, "depends_on": [7, 12],
 "reason": "Muon (exp 7) only works with arch 7 (exp 12)"}
```
This enables smart ablation (skip independent changes) and smart
crossover (combine only compatible changes).

---

## Phase 1 — The Core Loop

This is the heartbeat. Every experiment follows this exact protocol.

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
- Scale mutation magnitude with temperature using this guide:

```
T > 0.7  (hot)   → Structural changes. Swap algorithms, rewrite sections,
                    change architecture.
                    ML: replace optimizer (SGD→AdamW), change LR schedule
                    Code: change data structure (list→hashmap), new algorithm
                    Prompt: rewrite persona, reorganize all sections
                    Game: redesign resource system, change combat formula
T 0.3–0.7 (warm) → Parameter changes with moderate range.
                    ML: learning rate 3e-4→1e-3, depth 8→12
                    Code: buffer size 4K→16K, thread count 4→8
                    Prompt: add 2 examples, tighten constraints
                    Game: damage multiplier 1.0→1.5, XP curve exponent
T < 0.3  (cold)  → Fine-tuning. Small nudges to known-good values.
                    ML: learning rate 3e-4→2.5e-4, dropout 0.1→0.05
                    Code: cache TTL 300→360, batch size 100→120
                    Prompt: rephrase one sentence, adjust one example
                    Game: spawn rate 1.0→1.05, gold per kill ±5%
```

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
  "category": "architecture",
  "hypothesis": "Increasing depth from 8 to 10 layers should improve val_bpb because the model has more capacity for the fixed time budget",
  "mutation_description": "Changed DEPTH=8 to DEPTH=10 in train.py",
  "metric": 0.987,
  "previous_best": 0.993,
  "improvement_pct": 0.6,
  "status": "kept",
  "temperature": 0.31,
  "acceptance_probability": null,
  "duration_seconds": 302,
  "notes": "Third architecture change that improved. Depth seems to be the key lever."
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

### End-to-End Walkthrough — One Full Experiment

This shows experiment #7. The commands are domain-agnostic — substitute your
target file and metric. Examples: `train.py`/`val_bpb` (ML), `app.py`/`response_ms`
(code), `prompt.md`/`judge_score` (prompt), `balance.json`/`fairness_idx` (game).

```bash
# 1. Strategy Engine selects parameters
python strategy.py select
# Output: {"category": "architecture", "branch": "branch-1", "temperature": 0.38, "experiment_number": 7}

# 2. Agent reads current best on branch-1 and forms hypothesis
git checkout deepresearch/branch-1
cat ${TARGET_FILE} | head -50
cat .deepresearch/experiments.jsonl | tail -5  # Review recent experiments
# Hypothesis: "Changing X should improve ${METRIC} — experiments #3 and #5
# both improved with similar changes. T=0.38 supports moderate boldness."

# 3. Mutate: make ONE change to the target file
# (edit depends on domain — sed, patch, direct rewrite, etc.)

# 4. Execute eval harness with fixed budget
bash .deepresearch/eval.sh ${BUDGET} ${TARGET_FILE}
# Output: metric: <new_value>

# 5. Score + Decide
PREV_BEST=<branch_best>
NEW=<new_value>
# Compare using metric_direction from config.json

# 6a. IMPROVED → Keep
git add ${TARGET_FILE}
git commit -m "deepresearch #7: ${CATEGORY} — ${DESCRIPTION} (${METRIC}=${NEW})"

# 7. Update strategy state
python strategy.py update '{"id":7,"category":"...","branch":"branch-1","metric":${NEW},"previous_best":${PREV_BEST},"improvement_pct":...,"status":"kept","hypothesis":"...","mutation_description":"...","timestamp":"..."}'
# Output: [#7 | branch-1 | architecture | T=0.380] <metric> ✓ kept (+X.XX%)

# 8. → Loop back to step 1 for experiment #8
```

If the experiment had FAILED (new metric worse than previous best):
```bash
# Annealing check: P = exp(-|delta| / T)
# Example: P = exp(-|0.993-1.002| / 0.38) = exp(-0.024) = 0.976
# Roll random [0,1]. If roll < P → ACCEPT WORSE (annealing escape)
# If roll ≥ P → REJECT, revert:
git reset --hard HEAD~1
python strategy.py update '{"id":7,...,"status":"reverted",...}'
# Output: [#7 | branch-1 | architecture | T=0.380] <metric> ✗ reverted (-X.XX%)
```

---

## Phase 2 — Strategy Engine Details

### Intelligent Exploration Strategies

Beyond the basic bandit + annealing, use these meta-strategies:

**Momentum tracking:** If the last 3 experiments in a category all improved,
double down — try a BIGGER change in the same category.

```bash
# Check momentum before selecting next experiment
python3 -c "
import json
exps = [json.loads(l) for l in open('.deepresearch/experiments.jsonl') if l.strip()]
# Group last 3 experiments per category
from collections import defaultdict
recent = defaultdict(list)
for e in exps[-15:]:  # look at last 15
    recent[e.get('category','?')].append(e.get('status'))
for cat, statuses in recent.items():
    last3 = statuses[-3:]
    if len(last3) == 3 and all(s == 'kept' for s in last3):
        print(f'MOMENTUM: {cat} — 3 consecutive wins, go bigger')
    elif len(last3) == 3 and all(s in ('reverted','crashed') for s in last3):
        print(f'ANTI-MOMENTUM: {cat} — 3 consecutive failures, skip')
"
```

**Plateau detection:** If 5+ consecutive experiments show <0.1% change,
the agent is likely in a flat region. Trigger a reheat:

```bash
python3 -c "
import json
exps = [json.loads(l) for l in open('.deepresearch/experiments.jsonl') if l.strip()]
recent = exps[-5:]
if len(recent) >= 5 and all(abs(e.get('improvement_pct',0)) < 0.1 for e in recent):
    print('PLATEAU: 5 experiments with <0.1% change — reheat recommended')
"
```

Reheat action: set `temperature = min(current_T * 3, 0.8)` in strategy-state.json,
try a category unused in the last 10 experiments, and attempt a structural change.

**Regression analysis:** After 20+ experiments, analyze the log programmatically:

```bash
# Run after every 20 experiments
python3 -c "
import json, collections
exps = [json.loads(l) for l in open('.deepresearch/experiments.jsonl') if l.strip()]

# 1. Category success rates
cats = collections.defaultdict(lambda: {'kept':0,'total':0})
for e in exps:
    c = e.get('category','?')
    cats[c]['total'] += 1
    if e.get('status') == 'kept': cats[c]['kept'] += 1
for c, s in sorted(cats.items(), key=lambda x: -x[1]['kept']/max(x[1]['total'],1)):
    print(f'{c:20s} {s[\"kept\"]}/{s[\"total\"]} = {s[\"kept\"]/max(s[\"total\"],1)*100:.0f}%')

# 2. Sequential success pairs (A kept then B kept = interaction)
prev = None
pairs = collections.Counter()
for e in exps:
    if e.get('status') == 'kept' and prev:
        pairs[(prev, e['category'])] += 1
    prev = e.get('category') if e.get('status') == 'kept' else None
for (a,b), n in pairs.most_common(5):
    print(f'Pair: {a} → {b}: {n} co-successes')

# 3. Average improvement by category
import statistics
for c in cats:
    imps = [e['improvement_pct'] for e in exps if e.get('category')==c and e.get('status')=='kept' and 'improvement_pct' in e]
    if imps: print(f'{c}: avg improvement {statistics.mean(imps):.2f}%')
"
```

Use the output to adjust bandit priors and focus the next 10 experiments.

**Guided random restarts:** If stuck for 15+ experiments:

```bash
# 1. Save current best to population (it's already there via normal flow)

# 2. Identify top-3 most impactful commits
TOP3=$(python3 -c "
import json
exps = [json.loads(l) for l in open('.deepresearch/experiments.jsonl') if l.strip()]
kept = [e for e in exps if e.get('status')=='kept' and e.get('improvement_pct',0)>0]
kept.sort(key=lambda e: e['improvement_pct'], reverse=True)
for e in kept[:3]: print(e.get('id'))
")

# 3. Reset to baseline
git checkout deepresearch/branch-0 -- ${TARGET_FILES}

# 4. Cherry-pick only the top-3 changes
for ID in $TOP3; do
  HASH=$(git log --all --oneline --grep="deepresearch #${ID}:" | awk '{print $1}')
  [ -n "$HASH" ] && git cherry-pick --no-commit "$HASH"
done
git commit -m "deepresearch: guided restart with top-3 changes from experiments ${TOP3}"

# 5. Continue the loop from this cleaner starting point
# Temperature gets a reheat boost: T = min(current_T * 3, 0.8)
```

---

## Phase 3 — Persistent Memory

### Knowledge Base Schema

`.deepresearch/knowledge.json` accumulates insights across sessions:

```json
{
  "patterns": [
    {
      "domain": "ml_training",
      "category": "architecture",
      "description": "Increasing depth improves val_bpb up to ~12 layers, then diminishes",
      "confidence": 0.85,
      "evidence_count": 7,
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
      "domain": "ml_training",
      "category": "optimizer",
      "description": "Pure SGD without momentum always regresses vs Muon+AdamW baseline",
      "confidence": 0.95,
      "evidence_count": 4
    },
    {
      "domain": "code_optimization",
      "category": "parallelism",
      "description": "Adding threads to I/O-bound functions with GIL makes it slower, not faster",
      "confidence": 0.9,
      "evidence_count": 3
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
      "domain": "ml_training",
      "insight": "For 5-min budget on H100, model width matters more than depth beyond 8 layers",
      "source_session": "session-20260322",
      "metric_impact": "2.3% improvement"
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

After each experiment, run `python strategy.py update <result_json>` which
handles knowledge updates automatically. The logic:

```python
# Runs inside strategy.py update — excerpt of the key logic:

# 1. Pattern detection: 3+ successes in a category → record pattern
cat_successes = [e for e in exps if e['category'] == cat and e['status'] == 'kept']
if len(cat_successes) >= 3:
    knowledge['patterns'].append({
        'category': cat,
        'description': f'{cat} has {len(cat_successes)} successes',
        'confidence': len(cat_successes) / len(cat_exps)
    })

# 2. Anti-pattern: last 3 in category all failed → stop trying
recent_3 = [e for e in exps if e['category'] == cat][-3:]
if all(e['status'] in ('reverted','crashed') for e in recent_3):
    knowledge['anti_patterns'].append({
        'category': cat,
        'description': f'{cat} failed 3 consecutive times',
        'confidence': 0.8
    })

# 3. At session end: extract top insight
best_cat = max(arms, key=lambda c: arms[c]['alpha'] / (arms[c]['alpha'] + arms[c]['beta']))
knowledge['domain_insights'].append({
    'insight': f'{best_cat} was the most productive category this session',
    'source_session': session_id
})
```

At session START, `strategy.py select` automatically:
1. Loads knowledge.json and biases bandit arm priors from past success rates
2. Skips categories with high-confidence anti-patterns
3. Prints relevant insights for context

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

Alternatively, the agent can call the report logic directly:

```bash
python3 -c "
import json, datetime
exps = [json.loads(l) for l in open('.deepresearch/experiments.jsonl') if l.strip()]
state = json.load(open('.deepresearch/strategy-state.json'))
config = json.load(open('.deepresearch/config.json'))

total = len(exps)
kept = sum(1 for e in exps if e.get('status')=='kept')
crashed = sum(1 for e in exps if e.get('status')=='crashed')
baseline = state.get('baseline_metric', exps[0].get('metric') if exps else 'N/A')
best = state.get('best_metric', baseline)
direction = config.get('metric_direction','lower')
if baseline and best and isinstance(baseline,(int,float)):
    imp = ((baseline-best)/baseline*100) if direction=='lower' else ((best-baseline)/baseline*100)
else: imp = 0

# Top improvements
top = sorted([e for e in exps if e.get('status')=='kept' and e.get('improvement_pct',0)>0],
             key=lambda e: e['improvement_pct'], reverse=True)[:10]

now = datetime.datetime.now().strftime('%Y%m%d-%H%M')
report = f'''# DeepResearch Report
**Date:** {now} | **Experiments:** {total} ({kept} kept, {total-kept-crashed} reverted, {crashed} crashed)
**Baseline:** {baseline} | **Best:** {best} | **Improvement:** {imp:+.2f}%

## Top Changes
'''
for i,e in enumerate(top,1):
    report += f\"{i}. #{e['id']} ({e.get('category','?')}): {e.get('mutation_description',e.get('hypothesis','?'))} → {e.get('improvement_pct',0):+.2f}%\n\"

open(f'.deepresearch/reports/session-{now}.md','w').write(report)
print(f'Report saved to .deepresearch/reports/session-{now}.md')
"
```

### Report Template

Write to `.deepresearch/reports/session-YYYYMMDD-HHMM.md`:

```markdown
# DeepResearch Report — [Project/Domain]
**Session:** session-YYYYMMDD-HHMM
**Duration:** X hours, Y experiments
**Date:** YYYY-MM-DD

## Executive Summary
[2-3 sentences: what was optimized, how much it improved, key finding]

## Results
- **Baseline:** X.XXX
- **Final best:** X.XXX
- **Improvement:** +/-X.X%
- **Experiments run:** N total, N kept, N reverted, N crashed
- **Success rate:** X%

## Strategy Analysis

### Bandit Arm Performance
| Category | Trials | Successes | Success Rate | Avg Improvement |
|---|---|---|---|---|
| architecture | 12 | 5 | 42% | 0.8% |
| hyperparameters | 8 | 2 | 25% | 0.3% |
| ... | | | | |

### Temperature Profile
[How the temperature evolved, when reheats happened, annealing decisions]

### Population Diversity
[How branches diverged, crossover results, which branches survived]

## Top 10 Most Impactful Changes
1. **Experiment #N** (category): [description] → [metric change] [+/-X.X%]
2. ...

## Failed Approaches (Negative Results)
[These are equally valuable — document what DOESN'T work]
- [Approach]: tried N times, [why it fails]
- ...

## Discovered Patterns
[New entries added to knowledge.json during this session]

## Ablation Results
[Which changes in the best branch actually matter vs are noise]

## Recommendations for Next Session
1. [Most promising unexplored direction]
2. [Category worth deeper exploration]
3. [Suggested config changes for next run]

## Appendix: Full Experiment Log
[Reference to experiments.jsonl or condensed table]
```

---

## Domain Configurations

### ML Training (original autoresearch domain)

**Target:** train.py
**Metric:** val_bpb (lower is better)
**Budget:** 300 seconds (5 min wall clock)
**Mutation categories and example changes:**

- **architecture**: depth, width, attention patterns, activation functions,
  normalization, positional encoding, head count
- **hyperparameters**: learning rate, warmup steps, batch size, weight decay,
  gradient clipping, dropout
- **optimizer**: optimizer type, momentum, beta parameters, Muon vs AdamW mix,
  learning rate schedules
- **regularization**: dropout placement, weight tying, label smoothing,
  stochastic depth
- **data_processing**: sequence length, tokenizer vocab size, data sampling
  strategy, curriculum learning
- **scheduling**: LR schedule shape (cosine, linear, step), warmup duration,
  cooldown, cycle length
- **efficiency**: compilation flags, precision (bf16/fp16), memory optimization,
  gradient accumulation steps

### Code Performance Optimization

**Target:** specific source file(s)
**Metric:** benchmark time, memory usage, or composite
**Budget:** benchmark runtime + overhead (usually 30-120 seconds)
**Mutation categories:**

- **algorithm**: data structure changes, algorithmic complexity improvements,
  caching strategies
- **memory**: allocation patterns, pooling, reduce copies, in-place operations
- **parallelism**: threading, async, SIMD hints, batch processing
- **io**: buffering, lazy loading, serialization format
- **language_features**: compiler hints, type optimizations, inlining
- **architecture**: module decomposition, hot path optimization, code layout

### Prompt Engineering

**Target:** prompt template file
**Metric:** LLM-as-judge score (0-10) or task accuracy
**Budget:** N API calls per experiment (e.g., 10 test cases)
**Mutation categories:**

- **structure**: section ordering, heading hierarchy, instruction flow
- **specificity**: adding/removing constraints, examples, edge cases
- **tone**: formal/casual, assertive/collaborative, brief/detailed
- **examples**: few-shot examples, counter-examples, format demonstrations
- **guardrails**: boundary conditions, error handling instructions, refusal criteria
- **persona**: role definition, expertise level, communication style

### Game Balancing

**Target:** config/balance files
**Metric:** composite score (win rate variance, engagement proxy, fairness index)
**Budget:** N simulation rounds
**Mutation categories:**

- **economy**: resource generation rates, costs, trade ratios, inflation control
- **combat**: damage/health ratios, unit counters, speed/range tradeoffs
- **progression**: XP curves, unlock timing, power scaling
- **map_balance**: resource distribution, spawn fairness, strategic chokepoints
- **ai_behavior**: AI difficulty scaling, aggression parameters, decision weights

### Document Quality

**Target:** markdown/text document
**Metric:** LLM-as-judge rubric score
**Budget:** single LLM eval call per experiment
**Mutation categories:**

- **structure**: section ordering, hierarchy, information flow
- **clarity**: sentence length, jargon reduction, active voice
- **completeness**: missing topics, edge cases, examples
- **conciseness**: removing redundancy, tightening prose
- **formatting**: headers, lists, code blocks, emphasis

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
      model:"claude-sonnet-4-20250514", max_tokens:500,
      system: $p,
      messages:[{role:"user",content:$t}]
    }')" | jq -r '.content[0].text' | head -c 2000 | \
    curl -s https://api.anthropic.com/v1/messages \
    -H "Content-Type: application/json" -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d "$(jq -n --arg r "$(cat)" '{
      model:"claude-sonnet-4-20250514", max_tokens:50,
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

### State Validation (run at session start)

Before any experiment, validate all persistent state. Corrupted files are
the most common silent failure — catch them before they cascade.

```bash
# .deepresearch/validate.sh — run at session start
#!/bin/bash
set -e
ERRORS=0

# 1. Validate config.json
python3 -c "
import json, sys
c = json.load(open('.deepresearch/config.json'))
required = ['target_files','metric','metric_direction','budget_seconds','mutation_categories']
missing = [k for k in required if k not in c]
if missing: print(f'config.json missing keys: {missing}'); sys.exit(1)
if c['metric_direction'] not in ('lower','higher'): print('invalid metric_direction'); sys.exit(1)
" || { echo "✗ config.json corrupted"; ERRORS=$((ERRORS+1)); }

# 2. Validate strategy-state.json
python3 -c "
import json, sys
s = json.load(open('.deepresearch/strategy-state.json'))
required = ['temperature','total_experiments','bandit_arms','population']
missing = [k for k in required if k not in s]
if missing: print(f'strategy-state.json missing: {missing}'); sys.exit(1)
if not (0 <= s['temperature'] <= 2): print('temperature out of range'); sys.exit(1)
" || { echo "✗ strategy-state.json corrupted"; ERRORS=$((ERRORS+1)); }

# 3. Validate experiments.jsonl (each line must be valid JSON)
if [ -f .deepresearch/experiments.jsonl ]; then
  LINE=0
  while IFS= read -r line; do
    LINE=$((LINE+1))
    echo "$line" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null || {
      echo "✗ experiments.jsonl corrupted at line $LINE"
      ERRORS=$((ERRORS+1))
    }
  done < .deepresearch/experiments.jsonl
fi

# 4. Validate knowledge.json
python3 -c "
import json
k = json.load(open('.deepresearch/knowledge.json'))
assert isinstance(k.get('patterns'), list)
assert isinstance(k.get('anti_patterns'), list)
" 2>/dev/null || { echo "✗ knowledge.json corrupted"; ERRORS=$((ERRORS+1)); }

if [ $ERRORS -gt 0 ]; then
  echo "⚠ $ERRORS validation errors — run repair"
  exit 1
else
  echo "✓ All state files valid"
fi
```

### Corruption Repair

If validation fails, repair in this order:

1. **config.json corrupted** → Rebuild from git history:
   `git log --all --oneline -- .deepresearch/config.json` then restore,
   or re-run `init.sh` with the same domain settings.

2. **strategy-state.json corrupted** → Rebuild from experiments.jsonl:
   ```bash
   python3 -c "
   import json
   exps = [json.loads(l) for l in open('.deepresearch/experiments.jsonl') if l.strip()]
   arms = {}
   for e in exps:
       cat = e.get('category','unknown')
       if cat not in arms: arms[cat] = {'alpha':1,'beta':1,'trials':0}
       arms[cat]['trials'] += 1
       if e.get('status') in ('kept','accepted-worse'): arms[cat]['alpha'] += 1
       else: arms[cat]['beta'] += 1
   best = min((e['metric'] for e in exps if 'metric' in e and isinstance(e['metric'],(int,float))), default=None)
   state = {'temperature':0.5,'total_experiments':len(exps),'bandit_arms':arms,'population':[],'best_metric':best,'baseline_metric':exps[0]['metric'] if exps else None}
   json.dump(state, open('.deepresearch/strategy-state.json','w'), indent=2)
   print('✓ Rebuilt strategy-state.json from experiment log')
   "
   ```

3. **experiments.jsonl has bad lines** → Filter them out:
   ```bash
   python3 -c "
   import json
   good = []
   for line in open('.deepresearch/experiments.jsonl'):
       try: json.loads(line); good.append(line)
       except: pass
   open('.deepresearch/experiments.jsonl','w').writelines(good)
   print(f'✓ Kept {len(good)} valid experiment records')
   "
   ```

4. **knowledge.json corrupted** → Reset (knowledge is derived, not primary):
   `echo '{"patterns":[],"anti_patterns":[],"domain_insights":[],"cross_domain":[]}' > .deepresearch/knowledge.json`

### Backup Protocol

After every 10 experiments, create a timestamped state backup:
```bash
BACKUP=".deepresearch/backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP"
cp .deepresearch/config.json .deepresearch/strategy-state.json .deepresearch/knowledge.json "$BACKUP/"
cp .deepresearch/experiments.jsonl "$BACKUP/"
```

### Crash Recovery
If the agent itself crashes or the session is interrupted:
- `.deepresearch/experiments.jsonl` is append-only — no data loss
- `.deepresearch/strategy-state.json` is saved after every experiment
- Population snapshots are git-managed — always recoverable
- Next session: run `validate.sh`, repair if needed, then continue

### Git Safety
- Every experiment starts with `git stash` if there are uncommitted changes
- Every revert uses `git reset --hard` to a known-good commit
- The `.deepresearch/` directory is in `.gitignore` (state persists locally)
- Branch naming: `deepresearch/session-YYYYMMDD-HHMM`

### Rate Limiting (for LLM-as-judge evals)
- Track API usage in strategy-state.json
- If rate-limited, pause and retry with exponential backoff
- Reduce parallel agent count if hitting limits

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
✓ Mutate from BRANCH BEST, not from last attempt
✓ ONE change per experiment
✓ NEVER modify the evaluation harness
✓ NEVER ask "should I continue?" — run autonomously
✓ Log EVERYTHING — negative results are data
✓ Use the knowledge base — don't repeat known failures
✓ Temperature controls boldness — be bold early, precise late
✓ Crossover > random restart > greedy refinement
✓ Ablation keeps the branch clean — prune dead weight
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

### Troubleshooting — Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| First 10 experiments all reverted | Baseline is already near-optimal OR mutations too aggressive | Switch to `"conservative"` temperature, try finer-grained categories |
| Metric oscillates ±0.1% forever | Measurement noise exceeds mutation impact | Increase eval budget (more test cases, longer runs) to reduce variance |
| Every crossover crashes | Branches diverged structurally (incompatible changes) | Reset one branch to baseline + top-3, reduce divergence |
| Strategy always picks same category | One arm has high α from early luck, others starved | Manually reheat: set temperature to 0.8 in strategy-state.json |
| OOM crashes on every bold experiment | Model/buffer too large for hardware | Add a VRAM/memory check in eval.sh, reject before running |
| Metric is CRASHED but no error in log | Eval script doesn't print "metric:" line | Ensure eval.sh always ends with `echo "metric: $VALUE"` |
| Experiments run but knowledge.json empty | `strategy.py update` not called after each experiment | Must call `python strategy.py update '<json>'` after EVERY experiment |
| Resume starts from scratch | `.deepresearch/` was gitignored correctly but worktree was cleaned | Check `.deepresearch/` persists outside git worktrees |

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

This skill can be optimized using its own loop — target this SKILL.md,
score via composite quality across 3+ test optimization tasks from different
domains. The meta-loop: the research engine improving its own methodology.

---

## Implementation Notes for Claude Code

### Starting a Session

When the human says "run deepresearch" or similar:

1. Check if `.deepresearch/` exists
   - YES: Run `bash .deepresearch/validate.sh`, load state, show last report summary, ask "Resume or fresh start?"
   - NO: Run setup phase (0.1 through 0.3)
2. After confirmation, begin the core loop — **never ask again**
3. Print a one-line status after each experiment:
   `[#42 | branch-1 | architecture | T=0.31] metric: 0.987 → 0.981 ✓ kept (+0.6%)`
4. Use `grep`, `tail`, `head` — never `cat` on logs
5. Git integration: see walkthrough in Phase 1, crossover in Phase 1.7, parallel in Phase 4
