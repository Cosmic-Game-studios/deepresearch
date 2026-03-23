#!/usr/bin/env python3
"""
DeepResearch Realistic Benchmark — Simulates actual autoresearch scenarios.

Unlike continuous function optimization, autoresearch has:
1. CATEGORICAL mutations (architecture, optimizer, hyperparams — not continuous)
2. Different success rates per category (architecture works 40%, optimizer 10%)
3. Interaction effects (A+B together > A alone + B alone)
4. Local optima traps (greedy gets stuck when no single change helps)
5. Multi-session persistence (knowledge from session 1 helps session 2)

This benchmark simulates a "code optimization" scenario with 6 categories.
"""

import math
import random
import statistics
from dataclasses import dataclass, field

# ============================================================
# SIMULATED CODE OPTIMIZATION LANDSCAPE
# ============================================================

@dataclass
class CodeState:
    """Simulates the state of a codebase being optimized."""
    # Each dimension represents a "knob" the agent can turn
    architecture: int = 0      # 0-9: which architecture variant
    depth: int = 4             # 1-20: model depth
    lr: float = 0.001          # learning rate
    optimizer: int = 0         # 0=SGD, 1=Adam, 2=AdamW, 3=Muon
    batch_size: int = 32       # 8-512
    regularization: float = 0  # 0-1
    schedule: int = 0          # 0=constant, 1=cosine, 2=linear, 3=warmup_cosine

    def score(self) -> float:
        """
        Simulated metric (lower is better).
        Has multiple local optima, interaction effects, and one global optimum.
        """
        s = 100.0  # baseline

        # Architecture: variant 7 is best (-30), but variants 3-5 form a local trap (-15)
        arch_scores = {0: 0, 1: -5, 2: -8, 3: -15, 4: -16, 5: -15, 6: -10, 7: -30, 8: -12, 9: -5}
        s += arch_scores.get(self.architecture, 0)

        # Depth: optimal is 12, but 6-8 is a local optimum
        depth_score = -abs(self.depth - 12) * 1.5
        if 6 <= self.depth <= 8:
            depth_score = max(depth_score, -8)  # local optimum trap
        s += depth_score

        # Learning rate: optimal is 3e-4, but 1e-3 is a local optimum
        lr_opt = abs(math.log10(self.lr) - math.log10(3e-4))
        s += lr_opt * 10
        if 0.0008 < self.lr < 0.0012:
            s -= 3  # local trap at ~1e-3

        # Optimizer: Muon (3) is best but ONLY with architecture 7
        opt_scores = {0: 0, 1: -5, 2: -8, 3: -3}  # Muon seems worse alone
        s += opt_scores.get(self.optimizer, 0)

        # INTERACTION: Muon + arch 7 = massive bonus (-20 extra)
        if self.optimizer == 3 and self.architecture == 7:
            s -= 20

        # INTERACTION: cosine schedule + AdamW = bonus
        if self.schedule == 1 and self.optimizer == 2:
            s -= 8

        # INTERACTION: warmup_cosine + depth >= 10 = bonus
        if self.schedule == 3 and self.depth >= 10:
            s -= 12

        # Batch size: optimal is 128, quadratic penalty
        s += ((self.batch_size - 128) / 50) ** 2

        # Regularization: optimal is 0.1 when depth > 10
        if self.depth > 10:
            s += abs(self.regularization - 0.1) * 15
        else:
            s += self.regularization * 10  # regularization hurts shallow models

        # Schedule: warmup_cosine is best overall
        sched_scores = {0: 0, 1: -5, 2: -3, 3: -10}
        s += sched_scores.get(self.schedule, 0)

        # Add small noise (measurement noise)
        s += random.gauss(0, 0.5)

        return s

    def copy(self):
        return CodeState(
            self.architecture, self.depth, self.lr, self.optimizer,
            self.batch_size, self.regularization, self.schedule
        )

# Global optimum: arch=7, depth=12, lr=3e-4, optimizer=3(Muon), batch=128,
# reg=0.1, schedule=3(warmup_cosine) → score ≈ 100 - 30 - 18 - 0 - 3 - 20 - 12 - 0 - 0 - 10 = ~7

CATEGORIES = {
    "architecture": 0.35,    # 35% chance a random arch change helps
    "depth": 0.30,           # 30% chance
    "learning_rate": 0.25,   # 25% chance
    "optimizer": 0.15,       # 15% chance (Muon seems bad without arch 7)
    "batch_size": 0.20,      # 20% chance
    "regularization": 0.15,  # 15% chance
    "schedule": 0.25,        # 25% chance
}

def mutate_code(state: CodeState, category: str, bold: bool = False) -> CodeState:
    """Apply a categorical mutation."""
    s = state.copy()
    if category == "architecture":
        if bold:
            s.architecture = random.randint(0, 9)
        else:
            s.architecture = max(0, min(9, s.architecture + random.choice([-1, 1])))
    elif category == "depth":
        if bold:
            s.depth = random.randint(1, 20)
        else:
            s.depth = max(1, min(20, s.depth + random.choice([-2, -1, 1, 2])))
    elif category == "learning_rate":
        if bold:
            s.lr = 10 ** random.uniform(-5, -1)
        else:
            s.lr = s.lr * (10 ** random.uniform(-0.3, 0.3))
    elif category == "optimizer":
        s.optimizer = random.randint(0, 3)
    elif category == "batch_size":
        if bold:
            s.batch_size = random.choice([8, 16, 32, 64, 128, 256, 512])
        else:
            options = [s.batch_size // 2, s.batch_size * 2]
            s.batch_size = max(8, min(512, random.choice(options)))
    elif category == "regularization":
        if bold:
            s.regularization = random.uniform(0, 0.5)
        else:
            s.regularization = max(0, min(1, s.regularization + random.uniform(-0.05, 0.05)))
    elif category == "schedule":
        s.schedule = random.randint(0, 3)
    return s


# ============================================================
# STRATEGIES
# ============================================================

def run_greedy(n_experiments: int, seed: int):
    """Karpathy autoresearch: random category, keep only improvements."""
    random.seed(seed)
    state = CodeState()
    score = state.score()
    best_score = score
    history = [score]

    for _ in range(n_experiments):
        cat = random.choice(list(CATEGORIES.keys()))
        new_state = mutate_code(state, cat, bold=random.random() < 0.3)
        new_score = new_state.score()

        if new_score < score:
            state = new_state
            score = new_score
            best_score = min(best_score, score)

        history.append(best_score)

    return best_score, history

def run_bandit(n_experiments: int, seed: int):
    """Thompson sampling category selection, greedy acceptance."""
    random.seed(seed)
    state = CodeState()
    score = state.score()
    best_score = score
    history = [score]
    arms = {cat: {"a": 1, "b": 1} for cat in CATEGORIES}

    for _ in range(n_experiments):
        samples = {c: random.betavariate(a["a"], a["b"]) for c, a in arms.items()}
        cat = max(samples, key=samples.get)
        new_state = mutate_code(state, cat, bold=random.random() < 0.3)
        new_score = new_state.score()

        if new_score < score:
            state = new_state
            score = new_score
            best_score = min(best_score, score)
            arms[cat]["a"] += 1
        else:
            arms[cat]["b"] += 1

        history.append(best_score)

    return best_score, history

def run_deepresearch(n_experiments: int, seed: int):
    """Full DeepResearch: bandit + phased explore/exploit + population + adaptive."""
    random.seed(seed)

    K = 3
    branches = []
    for _ in range(K):
        s = CodeState()
        sc = s.score()
        branches.append({"state": s, "score": sc, "best_state": s.copy(), "best_score": sc})

    global_best = min(branches, key=lambda b: b["best_score"])
    best_score = global_best["best_score"]
    history = [best_score]

    arms = {cat: {"a": 1, "b": 1} for cat in CATEGORIES}
    no_improve = 0
    explore_end = int(n_experiments * 0.35)

    for i in range(n_experiments):
        exploring = i < explore_end

        # Boldness: high early, low late. Reheat if stuck.
        bold_prob = 0.5 if exploring else 0.15
        if no_improve >= 8:
            bold_prob = 0.6
            no_improve = 0

        # Thompson sampling (always)
        forced_explore = random.random() < (0.15 if exploring else 0.05)
        if forced_explore:
            cat = random.choice(list(CATEGORIES.keys()))
        else:
            samples = {c: random.betavariate(a["a"], a["b"]) for c, a in arms.items()}
            cat = max(samples, key=samples.get)

        # Branch selection: explore=spread, exploit=focus on best
        if exploring:
            bi = random.randint(0, K - 1)
        else:
            bi = min(range(K), key=lambda j: branches[j]["best_score"])
        branch = branches[bi]

        new_state = mutate_code(branch["state"], cat, bold=random.random() < bold_prob)
        new_score = new_state.score()

        if new_score < branch["score"]:
            branch["state"] = new_state
            branch["score"] = new_score
            arms[cat]["a"] += 1
            no_improve = 0
            if new_score < branch["best_score"]:
                branch["best_score"] = new_score
                branch["best_state"] = new_state.copy()
        elif exploring and (branch["score"] - new_score) > -3:
            # Conservative annealing: accept up to 3 points worse in explore phase
            branch["state"] = new_state
            branch["score"] = new_score
            arms[cat]["b"] += 1
            no_improve += 1
        else:
            arms[cat]["b"] += 1
            no_improve += 1

        # Crossover at phase boundary
        if i == explore_end:
            sb = sorted(branches, key=lambda b: b["best_score"])
            # Combine best two: take each attribute from the better branch
            # but try the other branch's attribute if it's from a high-success category
            hybrid = sb[0]["best_state"].copy()
            donor = sb[1]["best_state"]
            # Try swapping each attribute and test
            for attr in ["architecture", "depth", "lr", "optimizer", "batch_size", "regularization", "schedule"]:
                test = hybrid.copy()
                setattr(test, attr, getattr(donor, attr))
                if test.score() < hybrid.score():
                    setattr(hybrid, attr, getattr(donor, attr))
            hybrid_score = hybrid.score()
            worst_bi = max(range(K), key=lambda j: branches[j]["best_score"])
            if hybrid_score < branches[worst_bi]["best_score"]:
                branches[worst_bi] = {"state": hybrid, "score": hybrid_score,
                                      "best_state": hybrid.copy(), "best_score": hybrid_score}

        # Update global best
        for br in branches:
            if br["best_score"] < best_score:
                best_score = br["best_score"]

        history.append(best_score)

    return best_score, history


def run_deepresearch_multisession(n_experiments: int, seed: int, n_sessions: int = 3):
    """Multi-session DeepResearch: knowledge persists across sessions."""
    random.seed(seed)

    cumulative_arms = {cat: {"a": 1, "b": 1} for cat in CATEGORIES}
    best_score_ever = float('inf')
    best_state_ever = None
    total_history = []

    for session in range(n_sessions):
        per_session = n_experiments // n_sessions

        K = 3
        branches = []
        if best_state_ever and session > 0:
            # Session 2+: start one branch from best known state
            branches.append({"state": best_state_ever.copy(), "score": best_state_ever.score(),
                             "best_state": best_state_ever.copy(), "best_score": best_state_ever.score()})
        for _ in range(K - len(branches)):
            s = CodeState()
            sc = s.score()
            branches.append({"state": s, "score": sc, "best_state": s.copy(), "best_score": sc})

        arms = {cat: {"a": cumulative_arms[cat]["a"], "b": cumulative_arms[cat]["b"]}
                for cat in CATEGORIES}
        no_improve = 0
        explore_end = int(per_session * 0.3)

        for i in range(per_session):
            exploring = i < explore_end
            bold_prob = 0.4 if exploring else 0.1
            if no_improve >= 8:
                bold_prob = 0.5
                no_improve = 0

            if random.random() < (0.1 if exploring else 0.03):
                cat = random.choice(list(CATEGORIES.keys()))
            else:
                samples = {c: random.betavariate(a["a"], a["b"]) for c, a in arms.items()}
                cat = max(samples, key=samples.get)

            if exploring:
                bi = random.randint(0, K - 1)
            else:
                bi = min(range(K), key=lambda j: branches[j]["best_score"])
            branch = branches[bi]

            new_state = mutate_code(branch["state"], cat, bold=random.random() < bold_prob)
            new_score = new_state.score()

            if new_score < branch["score"]:
                branch["state"] = new_state
                branch["score"] = new_score
                arms[cat]["a"] += 1
                no_improve = 0
                if new_score < branch["best_score"]:
                    branch["best_score"] = new_score
                    branch["best_state"] = new_state.copy()
            else:
                arms[cat]["b"] += 1
                no_improve += 1

            for br in branches:
                if br["best_score"] < best_score_ever:
                    best_score_ever = br["best_score"]
                    best_state_ever = br["best_state"].copy()

            total_history.append(best_score_ever)

        # Persist arm knowledge across sessions
        cumulative_arms = {cat: {"a": arms[cat]["a"], "b": arms[cat]["b"]} for cat in CATEGORIES}

    return best_score_ever, total_history


# ============================================================
# BENCHMARK
# ============================================================

def main():
    N_EXP = 100
    N_RUNS = 50
    print(f"{'='*70}")
    print(f"  Realistic Autoresearch Benchmark — {N_EXP} experiments × {N_RUNS} runs")
    print(f"  Global optimum ≈ 7.0 (arch=7, depth=12, lr=3e-4, Muon, warmup_cos)")
    print(f"{'='*70}\n")

    strategies = {
        "GREEDY (autoresearch)": lambda n, s: run_greedy(n, s),
        "BANDIT only": lambda n, s: run_bandit(n, s),
        "DEEPRESEARCH (full)": lambda n, s: run_deepresearch(n, s),
        "DEEPRESEARCH (3 sessions)": lambda n, s: run_deepresearch_multisession(n, s, 3),
    }

    for name, fn in strategies.items():
        scores = []
        for seed in range(N_RUNS):
            score, _ = fn(N_EXP, seed)
            scores.append(score)

        mean = statistics.mean(scores)
        std = statistics.stdev(scores)
        median = statistics.median(scores)
        best = min(scores)

        print(f"  {name:32s}  mean={mean:7.2f} ± {std:5.2f}  "
              f"median={median:7.2f}  best={best:7.2f}")

    # Improvement summary
    print(f"\n{'─'*70}")
    greedy_scores = [run_greedy(N_EXP, s)[0] for s in range(N_RUNS)]
    deep_scores = [run_deepresearch(N_EXP, s)[0] for s in range(N_RUNS)]
    multi_scores = [run_deepresearch_multisession(N_EXP, s, 3)[0] for s in range(N_RUNS)]

    g_mean = statistics.mean(greedy_scores)
    d_mean = statistics.mean(deep_scores)
    m_mean = statistics.mean(multi_scores)

    print(f"  DeepResearch vs Greedy:          {(g_mean - d_mean) / g_mean * 100:+.1f}% improvement")
    print(f"  DeepResearch 3-session vs Greedy: {(g_mean - m_mean) / g_mean * 100:+.1f}% improvement")
    print(f"  (Global optimum ≈ 7.0, baseline ≈ 100)")


if __name__ == "__main__":
    main()
