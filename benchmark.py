#!/usr/bin/env python3
"""
DeepResearch Benchmark — Proves smart search beats greedy hill-climbing.

Tests 4 search strategies on optimization landscapes with known local optima:
1. GREEDY      — Karpathy autoresearch style (keep only improvements)
2. BANDIT      — Thompson sampling category selection (no annealing)
3. ANNEAL      — Greedy + simulated annealing (accept worse sometimes)
4. DEEPRESEARCH — Full stack: bandit + annealing + population + crossover

Each strategy gets 100 experiments on 3 test functions:
- Rastrigin: Many local optima (tests escape ability)
- Ackley: One global optimum surrounded by flat plateau (tests exploration)
- Schwefel: Deceptive — global optimum far from local optima (tests boldness)

Run:  python benchmark.py
"""

import math
import random
import json
import statistics
from dataclasses import dataclass, field
from typing import Callable

# ============================================================
# TEST FUNCTIONS (all have known global minima)
# ============================================================

def rastrigin(x: list[float]) -> float:
    """Many local optima. Global min = 0 at x=[0,0,...,0]"""
    A = 10
    n = len(x)
    return A * n + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)

def ackley(x: list[float]) -> float:
    """Flat plateau with narrow global min = 0 at x=[0,0,...,0]"""
    n = len(x)
    sum1 = sum(xi**2 for xi in x) / n
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x) / n
    return -20 * math.exp(-0.2 * math.sqrt(sum1)) - math.exp(sum2) + 20 + math.e

def schwefel(x: list[float]) -> float:
    """Deceptive. Global min ≈ 0 at x=[420.9687,...]. Second best is far away."""
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)


# ============================================================
# MUTATION CATEGORIES (simulate what an agent would try)
# ============================================================

CATEGORIES = ["shift_small", "shift_medium", "shift_large", "swap_dims", "reset_one"]

def mutate(x: list[float], category: str, temperature: float = 1.0) -> list[float]:
    """Apply a mutation to solution x."""
    x = x.copy()
    dim = random.randint(0, len(x) - 1)

    if category == "shift_small":
        x[dim] += random.gauss(0, 0.5 * temperature)
    elif category == "shift_medium":
        x[dim] += random.gauss(0, 2.0 * temperature)
    elif category == "shift_large":
        x[dim] += random.gauss(0, 10.0 * temperature)
    elif category == "swap_dims":
        dim2 = random.randint(0, len(x) - 1)
        x[dim], x[dim2] = x[dim2], x[dim]
    elif category == "reset_one":
        x[dim] = random.uniform(-500, 500) if random.random() < temperature else random.gauss(x[dim], 1.0)

    # Clamp to search space
    x = [max(-500, min(500, xi)) for xi in x]
    return x


# ============================================================
# SEARCH STRATEGIES
# ============================================================

@dataclass
class SearchState:
    best_x: list[float]
    best_score: float
    history: list[float] = field(default_factory=list)
    experiments: int = 0

def strategy_greedy(func: Callable, dims: int, n_experiments: int, seed: int) -> SearchState:
    """Karpathy autoresearch: keep only improvements, discard everything else."""
    random.seed(seed)
    x = [random.uniform(-5, 5) for _ in range(dims)]
    score = func(x)
    state = SearchState(best_x=x, best_score=score, history=[score])

    for _ in range(n_experiments):
        cat = random.choice(CATEGORIES)  # random category selection
        x_new = mutate(x, cat, temperature=1.0)  # no temperature decay
        score_new = func(x_new)
        state.experiments += 1

        if score_new < score:  # greedy: only keep improvements
            x = x_new
            score = score_new
            state.best_x = x
            state.best_score = score

        state.history.append(state.best_score)

    return state

def strategy_bandit(func: Callable, dims: int, n_experiments: int, seed: int) -> SearchState:
    """Thompson sampling for category selection, but still greedy acceptance."""
    random.seed(seed)
    x = [random.uniform(-5, 5) for _ in range(dims)]
    score = func(x)
    state = SearchState(best_x=x, best_score=score, history=[score])

    # Bandit arms
    arms = {cat: {"alpha": 1, "beta": 1} for cat in CATEGORIES}

    for _ in range(n_experiments):
        # Thompson sampling: sample from Beta, pick highest
        samples = {cat: random.betavariate(a["alpha"], a["beta"]) for cat, a in arms.items()}
        cat = max(samples, key=samples.get)

        x_new = mutate(x, cat, temperature=1.0)
        score_new = func(x_new)
        state.experiments += 1

        if score_new < score:
            x = x_new
            score = score_new
            state.best_x = x
            state.best_score = score
            arms[cat]["alpha"] += 1
        else:
            arms[cat]["beta"] += 1

        state.history.append(state.best_score)

    return state

def strategy_anneal(func: Callable, dims: int, n_experiments: int, seed: int) -> SearchState:
    """Greedy + simulated annealing (accept worse with probability)."""
    random.seed(seed)
    x = [random.uniform(-5, 5) for _ in range(dims)]
    score = func(x)
    state = SearchState(best_x=x[:], best_score=score, history=[score])
    best_ever_x = x[:]
    best_ever_score = score

    T_init = 0.5
    decay = 0.96
    score_range = abs(score) + 1e-6  # track range for normalization

    for i in range(n_experiments):
        T = T_init * (decay ** i)
        cat = random.choice(CATEGORIES)
        x_new = mutate(x, cat, temperature=max(T, 0.05))
        score_new = func(x_new)
        state.experiments += 1

        # Update score range estimate
        score_range = max(score_range, abs(score_new - best_ever_score))

        delta = score_new - score
        if delta < 0:  # improvement
            x = x_new
            score = score_new
        elif T > 0.01:
            # Normalize delta by score range so acceptance is scale-invariant
            norm_delta = abs(delta) / score_range
            p = math.exp(-norm_delta / T)
            if random.random() < p:
                x = x_new
                score = score_new

        if score < best_ever_score:
            best_ever_score = score
            best_ever_x = x[:]

        state.best_x = best_ever_x
        state.best_score = best_ever_score
        state.history.append(state.best_score)

    return state

def strategy_deepresearch(func: Callable, dims: int, n_experiments: int, seed: int) -> SearchState:
    """Full DeepResearch: bandit-first + conservative annealing + convergent population."""
    random.seed(seed)

    # Phase 1 (first 30%): K=3 branches, moderate exploration
    # Phase 2 (remaining): converge to best branch, exploit with bandit
    K = 3
    branches = []
    for _ in range(K):
        x = [random.uniform(-5, 5) for _ in range(dims)]
        s = func(x)
        branches.append({"x": x, "score": s, "best_x": x[:], "best_score": s})

    best_ever = min(branches, key=lambda b: b["best_score"])
    state = SearchState(
        best_x=best_ever["best_x"][:],
        best_score=best_ever["best_score"],
        history=[best_ever["best_score"]],
    )

    arms = {cat: {"alpha": 1, "beta": 1} for cat in CATEGORIES}
    consecutive_no_improve = 0
    explore_phase_end = int(n_experiments * 0.3)

    for i in range(n_experiments):
        in_explore_phase = i < explore_phase_end

        # Temperature: high in explore phase, low in exploit phase
        if in_explore_phase:
            T = 0.5 * (1 - i / explore_phase_end)  # 0.5 → 0 linearly
        else:
            T = 0.05  # nearly greedy in exploit phase

        # Adaptive reheat if stuck
        if consecutive_no_improve >= 12:
            T = 0.4
            consecutive_no_improve = 0

        # BANDIT: Thompson sampling (always — this is our strongest tool)
        if random.random() < T * 0.2:  # forced exploration only 10% at T=0.5
            cat = random.choice(CATEGORIES)
        else:
            samples = {c: random.betavariate(a["alpha"], a["beta"]) for c, a in arms.items()}
            cat = max(samples, key=samples.get)

        # Branch selection: explore phase uses all, exploit phase uses best only
        if in_explore_phase:
            branch_idx = random.randint(0, K - 1)
        else:
            branch_idx = min(range(K), key=lambda j: branches[j]["best_score"])
        branch = branches[branch_idx]

        x_new = mutate(branch["x"], cat, temperature=max(T, 0.05))
        score_new = func(x_new)
        state.experiments += 1

        delta = score_new - branch["score"]
        accepted = False

        if delta < 0:  # improvement — always accept
            accepted = True
            arms[cat]["alpha"] += 1
            consecutive_no_improve = 0
        elif in_explore_phase and delta > 0:
            # CONSERVATIVE annealing: only accept if delta is small relative to score
            relative_delta = abs(delta) / (abs(branch["score"]) + 1e-6)
            if relative_delta < 0.15 * T:  # at T=0.5 → accept up to 7.5% worse
                accepted = True
            arms[cat]["beta"] += 1
            consecutive_no_improve += 1
        else:
            arms[cat]["beta"] += 1
            consecutive_no_improve += 1

        if accepted:
            branch["x"] = x_new
            branch["score"] = score_new
            if score_new < branch["best_score"]:
                branch["best_score"] = score_new
                branch["best_x"] = x_new[:]

        # Crossover at phase transition: merge best traits of all branches
        if i == explore_phase_end and K >= 2:
            sorted_b = sorted(branches, key=lambda b: b["best_score"])
            for attempt in range(5):
                child = [sorted_b[0]["best_x"][d] * 0.8 + sorted_b[1]["best_x"][d] * 0.2
                         + random.gauss(0, 0.05) for d in range(dims)]
                child = [max(-500, min(500, xi)) for xi in child]
                child_score = func(child)
                if child_score < sorted_b[0]["best_score"]:
                    sorted_b[0]["x"] = child
                    sorted_b[0]["score"] = child_score
                    sorted_b[0]["best_x"] = child[:]
                    sorted_b[0]["best_score"] = child_score
                    break

        # Update global best
        for br in branches:
            if br["best_score"] < state.best_score:
                state.best_score = br["best_score"]
                state.best_x = br["best_x"][:]

        state.history.append(state.best_score)

    return state


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_benchmark(n_experiments: int = 100, n_runs: int = 30, dims: int = 5):
    """Run all strategies on all test functions, multiple seeds for statistics."""

    functions = {
        "Rastrigin (many local optima)": rastrigin,
        "Ackley (flat plateau)": ackley,
        "Schwefel (deceptive)": schwefel,
    }

    strategies = {
        "GREEDY (autoresearch)": strategy_greedy,
        "BANDIT only": strategy_bandit,
        "ANNEAL only": strategy_anneal,
        "DEEPRESEARCH (full)": strategy_deepresearch,
    }

    print(f"{'='*80}")
    print(f"DeepResearch Benchmark — {n_experiments} experiments × {n_runs} runs × {dims}D")
    print(f"{'='*80}\n")

    results = {}

    for func_name, func in functions.items():
        print(f"\n{'─'*60}")
        print(f"  {func_name}")
        print(f"{'─'*60}")

        results[func_name] = {}

        for strat_name, strat_fn in strategies.items():
            scores = []
            for seed in range(n_runs):
                state = strat_fn(func, dims, n_experiments, seed)
                scores.append(state.best_score)

            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0
            median = statistics.median(scores)
            best = min(scores)

            results[func_name][strat_name] = {
                "mean": mean, "std": std, "median": median, "best": best
            }

            print(f"  {strat_name:28s}  mean={mean:10.2f} ± {std:8.2f}  "
                  f"median={median:10.2f}  best={best:10.2f}")

    # Summary: improvement of DEEPRESEARCH over GREEDY
    print(f"\n{'='*80}")
    print(f"  SUMMARY: DeepResearch improvement over Greedy (autoresearch)")
    print(f"{'='*80}")

    for func_name in functions:
        greedy_mean = results[func_name]["GREEDY (autoresearch)"]["mean"]
        deep_mean = results[func_name]["DEEPRESEARCH (full)"]["mean"]
        improvement = (greedy_mean - deep_mean) / greedy_mean * 100
        print(f"  {func_name:40s}  {improvement:+.1f}% better")

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")

    return results


if __name__ == "__main__":
    run_benchmark()
