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

# DeepResearch — Autonomous Research Engine v2

An evolution of the Karpathy autoresearch pattern. The original insight stays:
hand the experiment loop to an AI agent and let it run autonomously. But where
autoresearch uses greedy hill-climbing on a single branch, DeepResearch adds
four layers of intelligence:

1. **Strategy Engine** — Bayesian bandit + simulated annealing replaces random exploration
2. **Population Search** — Multiple competing branches, not just one best
3. **Persistent Memory** — Knowledge survives across sessions and domains
4. **Research Reports** — Auto-generated analysis of findings

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
├── experiments.jsonl     # Append-only experiment log (one JSON per line)
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

**Budget per experiment:** Fixed time or resource budget. Every experiment gets
exactly the same budget so results are comparable. Default: 5 minutes.

**Population size K:** How many parallel branches to maintain. Default: 3.
Higher K = broader search but more overhead. For single-GPU ML: K=1 (sequential).
For code/prompt optimization: K=3-5 works well.

**Temperature schedule:** Controls exploration vs exploitation.
- `"aggressive"` — Start hot (T=1.0), cool slowly. Accept worse results early.
  Good when the search space is large and the baseline is far from optimal.
- `"moderate"` — Start warm (T=0.5), cool to 0.1 over 30 experiments.
  Good default for most problems.
- `"conservative"` — Start cool (T=0.2), only slight exploration.
  Good when the baseline is already well-optimized.

**Mutation categories:** Domain-specific list of change types the agent can try.
See Domain Configurations below. Each category becomes a "bandit arm."

### 0.2 Initialize

```bash
mkdir -p .deepresearch/populations/branch-0 .deepresearch/reports

# Save config
cat > .deepresearch/config.json << 'EOF'
{
  "target_files": ["train.py"],
  "metric": "val_bpb",
  "metric_direction": "lower",
  "budget_seconds": 300,
  "population_size": 3,
  "temperature_schedule": "moderate",
  "mutation_categories": [
    "architecture", "hyperparameters", "optimizer",
    "regularization", "data_processing", "scheduling"
  ],
  "created": "2026-03-23T12:00:00Z",
  "session_count": 0
}
EOF

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

## Phase 1 — The Core Loop

This is the heartbeat. Every experiment follows this exact protocol.

### 1.1 Select Strategy (Bandit + Temperature)

Before each experiment, the Strategy Engine decides:
- **WHICH category** to mutate (bandit selection)
- **HOW aggressively** to mutate (temperature)
- **WHICH branch** to mutate from (population selection)

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
- Scale the magnitude of the change with temperature:
  High T → bigger, bolder changes. Low T → small refinements.

### 1.4 Execute

Run the evaluation harness with fixed budget. Redirect ALL output.

```bash
# Run experiment with timeout
timeout ${BUDGET}s uv run train.py > .deepresearch/run.log 2>&1
EXIT_CODE=$?

# Extract metric
METRIC=$(grep "^${METRIC_NAME}:" .deepresearch/run.log | tail -1 | awk '{print $2}')
```

If the run crashes (empty metric, non-zero exit):
- Read the last 50 lines of the log for the error
- If the error is obvious (typo, import, OOM), try ONE fix
- If the fix also crashes, revert and log as "crashed"
- Move on — don't waste more than 2 attempts on a crash

### 1.5 Score + Decide

Compare the new metric to the CURRENT BRANCH BEST (not just the global best):

**Case: Improved** (lower val_bpb, higher accuracy, etc.)
→ Keep the change. Git commit. Update branch best.
→ If this branch now beats the global best, update global best.
→ Update bandit arm: α += 1 (success for this category)
→ Save to knowledge base as a successful pattern

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

### 1.7 Periodic Checkpoints

**Every 5 experiments:** Write a brief status update to the log:
- Current best metric vs baseline (absolute and %)
- Bandit arm statistics (which categories are winning)
- Temperature status
- Population diversity (how different are the branches)

**Every 10 experiments:** Trigger a CROSSOVER attempt:
- Take the top 2 branches
- Merge their changes intelligently (not blind file merge — understand what
  each branch changed and combine compatible changes)
- Test the crossover as a new experiment
- If it beats both parents, it becomes the new branch leader

**Every 20 experiments:** Perform ABLATION on the best branch:
- Systematically remove individual changes and re-test
- Identify which changes actually contribute and which are noise
- Prune unnecessary changes from the best branch
- This prevents "change accumulation" where many neutral changes pile up

### 1.8 Population Management

Maintain exactly K branches. When a new variant is born (from crossover or
a particularly good random exploration), decide which branch to replace:

- Never replace the current global best
- Replace the worst-performing branch
- If all branches are within 1% of each other, replace the OLDEST
  (stale branches are likely stuck in a local optimum)

---

## Phase 2 — Strategy Engine Details

### Intelligent Exploration Strategies

Beyond the basic bandit + annealing, use these meta-strategies:

**Momentum tracking:** If the last 3 experiments in a category all improved,
double down — try a BIGGER change in the same category. The search landscape
is likely smooth in this direction.

**Plateau detection:** If 5+ consecutive experiments show <0.1% change
(in either direction), the agent is likely in a flat region:
- Increase temperature temporarily (reheat)
- Try a category that hasn't been tried in 10+ experiments
- Attempt a structural change rather than a parameter tweak

**Regression analysis:** After 20+ experiments, look for correlations:
- Which combinations of categories tend to succeed together?
- Is there a relationship between change magnitude and improvement?
- Are there interaction effects (A works only when B is also present)?

**Guided random restarts:** If stuck for 15+ experiments:
- Save the current best to the population
- Reset the working branch to baseline
- Apply ONLY the top-3 most impactful changes from the log
- Continue from this cleaner starting point

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
    }
  ],
  "anti_patterns": [
    {
      "domain": "ml_training",
      "category": "optimizer",
      "description": "Pure SGD without momentum always regresses vs Muon+AdamW baseline",
      "confidence": 0.95,
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

After each experiment:
1. If a pattern repeats 3+ times → add/update in `patterns`
2. If something fails 3+ times → add to `anti_patterns`
3. At session end → extract `domain_insights` from the experiment log

At session START:
1. Load knowledge.json
2. Use patterns to BIAS initial bandit arm priors (start arms with α/β
   reflecting past success rates, not uniform)
3. Use anti_patterns to SKIP known-bad approaches
4. Use domain_insights to form better initial hypotheses

### Cross-Session Continuity

When starting a new session in the same project:
1. Load `.deepresearch/strategy-state.json` — resume temperature, bandit stats
2. Load `.deepresearch/experiments.jsonl` — full history
3. Load population snapshots — continue from the best branches
4. Increment session_count in config.json

The agent should read the last session's report (if any) and use it as
context for forming the first hypothesis of the new session.

---

## Phase 4 — Multi-Agent (Claude Code Subagents)

When the environment supports parallel execution (e.g., multiple GPUs, or
code/prompt optimization where eval is fast), spawn parallel agents.

### Parallel Architecture

```
Main Agent (Orchestrator)
├── Agent α — Branch 0 (exploitation: refine current best)
├── Agent β — Branch 1 (exploration: bold category changes)
├── Agent γ — Branch 2 (crossover: combine best of α and β)
└── Agent δ — Branch 3 (random restart: fresh perspective)
```

### Orchestration Protocol

The main agent:
1. Assigns each subagent a branch and a role (exploit/explore/crossover/restart)
2. Waits for all agents to complete one experiment each
3. Collects results
4. Updates the global strategy state (bandit arms, temperature, population)
5. Reassigns branches and roles for the next round
6. Handles crossover/population management centrally

Each subagent:
1. Receives: branch files, experiment history, knowledge base, assigned category
2. Runs one experiment following the core loop (mutate, execute, score)
3. Returns: metric, git diff, hypothesis, result status

### When to Parallelize

- **ML training on multiple GPUs**: Each agent trains on a different GPU.
  Set `CUDA_VISIBLE_DEVICES` per agent.
- **Code optimization**: Eval is usually fast (<30s). Run 3-5 agents in
  parallel, each testing a different mutation.
- **Prompt optimization with LLM-as-judge**: API calls can be parallelized.
  Run multiple prompt variants through the judge concurrently.
- **Game balancing**: If simulation is fast, run different parameter
  combinations in parallel.

### Sequential Fallback

If parallel execution isn't available (single GPU, rate-limited API), the
orchestrator runs agents sequentially but still maintains the population
and bandit statistics. The search strategy remains the same — just slower.

---

## Phase 5 — Research Reports

### Auto-Generation Triggers

Generate a report when:
- A session ends (user interrupt, target reached, or context limit)
- Every 25 experiments within a session (progress report)
- A significant breakthrough occurs (>5% improvement in one experiment)

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

When stopping for ANY reason:
1. Generate the full research report
2. Save strategy state (so next session can resume)
3. Update knowledge base with session findings
4. Commit the best state on the main branch
5. Present the report to the human

---

## Error Recovery

### Crash Recovery
If the agent itself crashes or the session is interrupted:
- `.deepresearch/experiments.jsonl` is append-only — no data loss
- `.deepresearch/strategy-state.json` is saved after every experiment
- Population snapshots are git-managed — always recoverable
- Next session: load all state and continue

### Git Safety
- Every experiment starts with `git stash` if there are uncommitted changes
- Every revert uses `git reset --hard` to a known-good commit
- The `.deepresearch/` directory is in `.gitignore` (state persists locally)
- Branch naming: `deepresearch/session-YYYYMMDD-HHMM`

### Rate Limiting (for LLM-as-judge evals)
- Track API usage in strategy-state.json
- If rate-limited, pause and retry with exponential backoff
- Reduce parallel agent count if hitting limits

---

## Quick Reference

```
SETUP:
  1. What artifact?      → Files being optimized
  2. What metric?        → ONE number, direction clear
  3. What harness?       → Immutable evaluation (NEVER modified)
  4. What budget?        → Fixed per experiment
  5. What population?    → K parallel branches (default 3)
  6. What temperature?   → aggressive / moderate / conservative

STRATEGY ENGINE:
  • Bandit Selection    → Thompson sampling over mutation categories
  • Temperature         → Simulated annealing (accept worse early, greedy late)
  • Population          → Top-K branches, tournament selection, crossover

CORE LOOP:
  1. Select strategy   (bandit + temperature + branch)
  2. Form hypothesis   (informed by history + knowledge base)
  3. Mutate            (one focused change, magnitude ∝ temperature)
  4. Execute           (fixed budget, redirect output)
  5. Score + Decide    (improved? keep. worse? anneal or revert.)
  6. Log everything    (experiments.jsonl — wins AND losses)
  7. Update memory     (patterns, anti-patterns, knowledge)
  8. → back to 1

CHECKPOINTS:
  • Every 5:  status update
  • Every 10: crossover attempt
  • Every 20: ablation analysis
  • Every 25: progress report

PRINCIPLES:
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

## Self-Improvement

This skill can be optimized using its own loop:
- **Target:** This SKILL.md
- **Metric:** Composite score across 3+ test optimization tasks
- **Eval:** Run DeepResearch on a prompt, a code file, and a config.
  Score how well the loop performs (improvement achieved / experiments needed).

The meta-loop: the research engine improving its own research methodology.

---

## Implementation Notes for Claude Code

### Starting a Session

When the human says "run deepresearch" or similar:

1. Check if `.deepresearch/` exists
   - YES: Load state, show last session summary, ask "Resume or fresh start?"
   - NO: Run setup phase (0.1 through 0.3)
2. After setup/resume confirmation, begin the core loop
3. Print a one-line status after each experiment:
   `[#42 | branch-1 | architecture | T=0.31] val_bpb: 0.987 → 0.981 ✓ kept (+0.6%)`
4. Never flood context — redirect experiment output to files
5. Read only what you need from logs (grep, tail, head)

### Git Integration

```bash
# Create session branch
git checkout -b deepresearch/session-$(date +%Y%m%d-%H%M)

# On kept experiment
git add ${TARGET_FILES}
git commit -m "deepresearch #${ID}: ${CATEGORY} — ${DESCRIPTION} (${METRIC})"

# On reverted experiment
git reset --hard HEAD

# On crossover
git checkout -b deepresearch/crossover-${ID}
# ... apply merged changes ...
# If better, merge into the winning branch
```

### Subagent Spawning (for parallel mode)

In Claude Code, use `claude --print` or background processes:
```bash
# Spawn parallel experiments (pseudo-code)
for branch in branch-0 branch-1 branch-2; do
  (
    cd /tmp/deepresearch-${branch}
    git checkout deepresearch/${branch}
    # Apply mutation
    # Run eval
    # Write result to .deepresearch/results-${branch}.json
  ) &
done
wait
# Orchestrator collects results
```

Adapt to your actual Claude Code subagent capabilities.
