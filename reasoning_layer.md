# Reasoning Layer — The Intelligence Amplifier
# Add this section to SKILL.md after Phase 1 Core Loop

## The Reasoning Layer (what makes this different)

Every other autoresearch tool treats the LLM as a code editor that follows
bandit statistics. DeepResearch treats it as a **thinking researcher** that
uses statistics as ONE input alongside understanding, theory, and reflection.

This is the core differentiator. Opus 4.6 can reason about WHY experiments
work, see patterns humans miss, form causal theories, and predict outcomes.
The Reasoning Layer has three components that wrap around every experiment.

### R1: Deep Read (before each experiment)

Before mutating, THINK about the artifact. Don't just grep for numbers.

```
DEEP READ PROTOCOL:
1. Read the target file(s) completely. Understand what each section does.
2. Identify the BOTTLENECK — what is currently limiting the metric?
   Not "what could be changed" but "what is the weakest link RIGHT NOW."
3. Check experiment history: has this bottleneck been addressed before?
   If yes, why didn't it work? What's different about your approach?
4. Check the knowledge base: any patterns or anti-patterns relevant?
5. Form a CAUSAL MODEL: "The metric is X because of Y, and changing Z
   should improve it because [specific mechanism]."
```

The Deep Read should take ~20% of your thinking time per experiment.
This is NOT wasted time — it's what separates intelligent search from
random search. A well-formed hypothesis has a 3-5x higher success rate
than a random mutation.

**Opus 4.6 advantage:** Use adaptive thinking here. The model naturally
thinks deeper on complex artifacts. Don't rush — let it reason about
the code structure, data flow, and performance characteristics.

### R2: Causal Hypothesis (replaces "pick a category")

The bandit statistics INFORM but don't DICTATE. The agent forms a
hypothesis that combines:

```
HYPOTHESIS = f(
  Deep Read understanding,     # What the code actually does
  Experiment history,          # What worked before and why
  Bandit statistics,           # Which categories have high success rates
  Knowledge base patterns,     # Cross-session insights
  Domain knowledge,            # What the model knows about this domain
  Temperature                  # How bold to be
)
```

Write the hypothesis as a structured thought BEFORE making any changes:

```markdown
## Hypothesis for Experiment #N

**Theory:** The metric is currently limited by [specific bottleneck]
because [causal explanation].

**Prediction:** Changing [specific thing] should improve the metric
by approximately [magnitude] because [mechanism].

**Connects to:** This builds on experiment #X which showed [finding].
If experiment #Y's [approach] failed because [reason], this avoids
that pitfall by [difference].

**Risk:** This could fail if [condition]. Fallback: [alternative].

**Category:** [mutation category] (bandit success rate: X%)
**Confidence:** [low/medium/high]
```

The key shift: autoresearch says "bandit picked 'architecture', so change
something architectural." DeepResearch says "I've read the code, the
bottleneck is that the attention computation is quadratic in sequence
length, and based on experiments #5 and #8 showing that efficiency changes
compound, I should try windowed attention — this is an 'architecture'
change but the REASON matters more than the category."

### R3: Reflection (after each experiment)

After seeing the result, DON'T just update a counter. THINK about it:

```
REFLECTION PROTOCOL:
1. Was my prediction correct? Did the metric move in the direction
   and magnitude I expected?
2. If YES: What does this confirm about my causal model? What's the
   next logical experiment to push this further?
3. If NO: Why was I wrong? Three possibilities:
   a. My theory was wrong (need new theory)
   b. My theory was right but the implementation was off (try again differently)
   c. There's an interaction effect I didn't account for (update model)
4. What did I LEARN that I didn't know before? Write this down.
5. Does this change my understanding of the bottleneck? Has the
   bottleneck shifted to something else now?
```

Write a 2-3 sentence reflection in the experiment log. Examples:

GOOD reflection:
"Experiment #12 confirmed that depth scaling follows diminishing returns
past 12 layers (predicted ~0.5% improvement, got 0.1%). The bottleneck
has shifted from capacity to optimization — next experiments should
target learning rate schedule and warmup."

BAD reflection:
"Metric improved. Kept."

### Research Memos (every 10 experiments)

Every 10 experiments, write a research memo in `.deepresearch/memos/`:

```markdown
# Research Memo — Experiments #N to #N+10

## Current Theory
[What you believe is the dominant factor affecting the metric right now]

## What We Learned
[Key findings from the last 10 experiments]

## Causal Graph Update
[How your understanding of cause→effect has changed]
Example: "Previously thought optimizer choice was independent of architecture.
Experiments #15-18 showed Muon only works with arch variant 7 — there's a
strong interaction. Updated knowledge base."

## Dead Ends (stop trying)
[Approaches that are confirmed to not work, and WHY]

## Most Promising Direction
[Where the next 10 experiments should focus, and why]

## Confidence Assessment
[How confident are you in your current theory? What would change your mind?]
```

These memos are the most valuable output of the entire system. They're
the "paper" that would convince another researcher (or a future session)
of your findings. An experiment log without memos is just noise.

**Opus 4.6 advantage:** The 1M context window means the agent can hold
ALL previous memos in context simultaneously. It can see connections
across 100+ experiments that no human would catch. This is where the
model's intelligence compounds — memo #5 references memo #3 which
referenced memo #1, creating a coherent research narrative.

### Causal Dependency Tracking

Don't just track "what changed" — track "what depends on what."

```json
{
  "dependencies": [
    {
      "experiment": 15,
      "depends_on": [7, 12],
      "reason": "Muon optimizer (exp 7) only works because we switched to arch 7 (exp 12). Reverting exp 12 would likely break exp 15's improvement."
    },
    {
      "experiment": 22,
      "independent": true,
      "reason": "Batch size change is orthogonal to architecture. Can be applied to any branch."
    }
  ]
}
```

This enables SMART ablation: instead of testing every commit independently,
test only experiments marked as potentially dependent. If experiment #22
is marked independent, skip it in ablation — it's safe.

This also enables SMART crossover: when combining branches, prioritize
independent changes (safe to combine) and be cautious with dependent
chains (might conflict).

### Connecting It All: The Thinking Researcher Loop

```
┌─────────────────────────────────────────────────┐
│  1. DEEP READ   — Understand the artifact       │
│  2. THEORIZE    — Form causal hypothesis        │
│  3. PREDICT     — What should happen and why    │
│  4. MUTATE      — One change to test the theory │
│  5. EXECUTE     — Fixed budget, redirect output │
│  6. COMPARE     — Was prediction correct?       │
│  7. REFLECT     — Why? Update causal model      │
│  8. LOG         — Hypothesis + result + learning│
│  9. MEMO (×10)  — Synthesize into research memo │
│ 10. REPEAT      — Next hypothesis from new model│
└─────────────────────────────────────────────────┘
```

The difference: steps 1-3 and 6-7 are THINKING steps that use the full
power of the model's reasoning. Autoresearch only does steps 4-5-8.
Those thinking steps are what make the search intelligent rather than
random.

### Why This Works Better (the math)

Consider 100 experiments. With greedy autoresearch:
- ~30% of mutations are in productive categories (random selection)
- ~10-15% of experiments produce improvements
- No learning between experiments → flat improvement rate

With DeepResearch's Reasoning Layer:
- After 20 experiments, the agent understands the bottleneck
- ~60-70% of mutations target the actual bottleneck (informed selection)
- ~25-35% of experiments produce improvements
- Learning compounds: each experiment informs the next → accelerating rate

The Reasoning Layer doesn't replace the bandit — it makes the bandit
smarter. The bandit tracks WHICH categories work. The Reasoning Layer
understands WHY they work and WHEN they'll work again.
