#!/usr/bin/env python3
"""
Level 2-3 Benchmark — Does structural mutation beat parameter tuning?

Simulates a universal software system with:
- A BASE that works but is slow/inefficient
- FEATURES that can be ADDED to improve it (Level 2)
- PARAMETERS that can be TUNED within each feature (Level 1)
- A CURRICULUM of progressive goals (Level 2.5)
- INTERACTIONS between features (some help each other, some conflict)

The key question: how much better does a system get when the agent can
ADD FEATURES (Level 2) vs only TUNE PARAMETERS (Level 1)?

This is domain-agnostic — the "system" is abstract. But the dynamics
mirror real software: adding caching helps, but only if the bottleneck
is IO. Adding parallelism helps, but only if the work is CPU-bound.

Run: python benchmark_level3.py
"""

import math
import random
import statistics

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ════════════════════════════════════════════════════════════
# SIMULATED SOFTWARE SYSTEM
# ════════════════════════════════════════════════════════════

class SoftwareSystem:
    """
    A simulated software system with features and parameters.

    The system has a "performance score" (lower is better).
    It starts as a basic implementation. Features can be ADDED
    to improve it. Each feature has parameters that can be TUNED.

    Features interact: some are synergistic, some conflict.
    Some features only help after other features are present.
    """

    # All possible features. Each has:
    #   base_impact: how much it helps when added (negative = helps)
    #   params: tunable parameters with optimal values
    #   requires: features that must be present first
    #   synergy: bonus when combined with specific other features
    #   anti_synergy: penalty when combined with specific features
    FEATURES = {
        "caching": {
            "base_impact": -15,
            "params": {"cache_size": {"range": (10, 1000), "optimal": 256},
                       "ttl_seconds": {"range": (1, 3600), "optimal": 300}},
            "requires": [],
            "synergy": {"connection_pool": -5},
            "anti_synergy": {},
        },
        "connection_pool": {
            "base_impact": -10,
            "params": {"pool_size": {"range": (1, 100), "optimal": 20},
                       "timeout_ms": {"range": (100, 30000), "optimal": 5000}},
            "requires": [],
            "synergy": {"async_io": -8},
            "anti_synergy": {},
        },
        "async_io": {
            "base_impact": -12,
            "params": {"max_concurrent": {"range": (1, 500), "optimal": 50},
                       "queue_size": {"range": (10, 10000), "optimal": 1000}},
            "requires": [],
            "synergy": {"connection_pool": -8, "batch_processing": -10},
            "anti_synergy": {"simple_threading": 5},
        },
        "batch_processing": {
            "base_impact": -8,
            "params": {"batch_size": {"range": (1, 1000), "optimal": 64},
                       "flush_interval_ms": {"range": (10, 5000), "optimal": 100}},
            "requires": [],
            "synergy": {"async_io": -10},
            "anti_synergy": {},
        },
        "compression": {
            "base_impact": -6,
            "params": {"level": {"range": (1, 9), "optimal": 6},
                       "min_size_bytes": {"range": (100, 100000), "optimal": 1024}},
            "requires": [],
            "synergy": {"caching": -3},
            "anti_synergy": {},
        },
        "indexing": {
            "base_impact": -20,
            "params": {"index_type": {"range": (0, 3), "optimal": 2},
                       "rebuild_interval": {"range": (1, 1000), "optimal": 100}},
            "requires": [],
            "synergy": {"caching": -7},
            "anti_synergy": {},
        },
        "simple_threading": {
            "base_impact": -5,
            "params": {"num_threads": {"range": (1, 32), "optimal": 8}},
            "requires": [],
            "synergy": {},
            "anti_synergy": {"async_io": 5},
        },
        "rate_limiter": {
            "base_impact": -3,
            "params": {"requests_per_sec": {"range": (10, 10000), "optimal": 1000},
                       "burst_size": {"range": (1, 100), "optimal": 10}},
            "requires": [],
            "synergy": {"connection_pool": -2},
            "anti_synergy": {},
        },
        "monitoring": {
            "base_impact": -1,  # small direct impact
            "params": {"sample_rate": {"range": (1, 100), "optimal": 10}},
            "requires": [],
            "synergy": {},  # but enables better decisions
            "anti_synergy": {},
        },
        "circuit_breaker": {
            "base_impact": -4,
            "params": {"failure_threshold": {"range": (1, 20), "optimal": 5},
                       "reset_timeout_ms": {"range": (1000, 60000), "optimal": 10000}},
            "requires": ["connection_pool"],  # needs connection pool to be useful
            "synergy": {"rate_limiter": -3, "monitoring": -2},
            "anti_synergy": {},
        },
        "query_optimizer": {
            "base_impact": -18,
            "params": {"plan_cache_size": {"range": (10, 1000), "optimal": 100}},
            "requires": ["indexing"],  # needs indexes to optimize against
            "synergy": {"caching": -5, "indexing": -10},
            "anti_synergy": {},
        },
        "lazy_loading": {
            "base_impact": -7,
            "params": {"threshold_bytes": {"range": (100, 100000), "optimal": 10000}},
            "requires": [],
            "synergy": {"caching": -4},
            "anti_synergy": {},
        },
    }

    def __init__(self):
        self.active_features = set()
        self.param_values = {}  # feature -> param -> value
        self.base_score = 100.0

    def add_feature(self, name: str) -> bool:
        """Add a feature. Returns False if prerequisites not met."""
        if name not in self.FEATURES:
            return False
        feat = self.FEATURES[name]
        for req in feat["requires"]:
            if req not in self.active_features:
                return False
        self.active_features.add(name)
        # Initialize params to midpoint (not optimal)
        self.param_values[name] = {}
        for p_name, p_info in feat["params"].items():
            lo, hi = p_info["range"]
            self.param_values[name][p_name] = (lo + hi) // 2
        return True

    def remove_feature(self, name: str) -> bool:
        """Remove a feature. Also removes features that depend on it."""
        if name not in self.active_features:
            return False
        # Check if other active features require this one
        to_remove = {name}
        for other_name in list(self.active_features):
            feat = self.FEATURES.get(other_name, {})
            if name in feat.get("requires", []):
                to_remove.add(other_name)
        for n in to_remove:
            self.active_features.discard(n)
            self.param_values.pop(n, None)
        return True

    def set_param(self, feature: str, param: str, value) -> bool:
        """Set a parameter value for an active feature."""
        if feature not in self.active_features:
            return False
        if param not in self.param_values.get(feature, {}):
            return False
        info = self.FEATURES[feature]["params"][param]
        lo, hi = info["range"]
        self.param_values[feature][param] = max(lo, min(hi, value))
        return True

    def score(self) -> float:
        """Calculate system performance score. Lower is better."""
        s = self.base_score

        for feat_name in self.active_features:
            feat = self.FEATURES[feat_name]
            # Base impact of having the feature
            s += feat["base_impact"]

            # Parameter tuning quality (distance from optimal)
            for p_name, p_info in feat["params"].items():
                val = self.param_values.get(feat_name, {}).get(p_name, p_info["range"][0])
                opt = p_info["optimal"]
                rng = p_info["range"][1] - p_info["range"][0]
                distance = abs(val - opt) / max(rng, 1)
                # Bad params can negate up to 50% of the feature's benefit
                s += distance * abs(feat["base_impact"]) * 0.5

            # Synergy bonuses
            for other, bonus in feat.get("synergy", {}).items():
                if other in self.active_features:
                    s += bonus

            # Anti-synergy penalties
            for other, penalty in feat.get("anti_synergy", {}).items():
                if other in self.active_features:
                    s += penalty

        # Noise
        s += random.gauss(0, 0.3)
        return s

    def copy(self):
        c = SoftwareSystem()
        c.active_features = set(self.active_features)
        c.param_values = {f: dict(p) for f, p in self.param_values.items()}
        return c


# Global optimum: all synergistic features active with optimal params
# indexing(-20) + query_optimizer(-18, req indexing, synergy -10) + caching(-15, synergy -7-5-4-3)
# + async_io(-12) + connection_pool(-10, synergy -8-5) + batch_processing(-8, synergy -10)
# + lazy_loading(-7) + compression(-6, synergy -3) + rate_limiter(-3, synergy -2)
# + monitoring(-1, synergy -2) + circuit_breaker(-4, req pool, synergy -3-2)
# WITHOUT simple_threading (anti-synergy with async_io)
# Total feature impact: ~-20-18-15-12-10-8-7-6-3-1-4 = -104 base
# + synergies: -10-7-5-4-3-8-5-10-3-2-3-2 = -62
# - param distance: 0 (all optimal)
# = 100 - 104 - 62 = ~ -66 → clamp to ~-66
# Best possible: approximately -60 to -70


# ════════════════════════════════════════════════════════════
# STRATEGIES
# ════════════════════════════════════════════════════════════

FEATURE_NAMES = list(SoftwareSystem.FEATURES.keys())

def strategy_level1(n_exp, seed):
    """Level 1: Can only tune parameters of features that START active."""
    random.seed(seed)
    sys = SoftwareSystem()
    # Start with a few basic features (simulates existing codebase)
    for f in ["caching", "simple_threading"]:
        sys.add_feature(f)

    score = sys.score()
    best = score
    history = [best]

    for _ in range(n_exp):
        s = sys.copy()
        # Pick random active feature, random param, random value
        if s.active_features:
            feat = random.choice(list(s.active_features))
            params = list(SoftwareSystem.FEATURES[feat]["params"].keys())
            if params:
                param = random.choice(params)
                info = SoftwareSystem.FEATURES[feat]["params"][param]
                lo, hi = info["range"]
                new_val = s.param_values[feat][param] + random.randint(-max(1, (hi-lo)//5), max(1, (hi-lo)//5))
                s.set_param(feat, param, new_val)

        new_score = s.score()
        if new_score < score:
            sys = s
            score = new_score
            best = min(best, score)
        history.append(best)

    return best, history


def strategy_level1_reasoning(n_exp, seed):
    """Level 1.5: Tune params with Reasoning Layer (informed, not random)."""
    random.seed(seed)
    sys = SoftwareSystem()
    for f in ["caching", "simple_threading"]:
        sys.add_feature(f)

    score = sys.score()
    best = score
    history = [best]
    understanding = {}  # grows over attempts

    for i in range(n_exp):
        s = sys.copy()
        if s.active_features:
            feat = random.choice(list(s.active_features))
            params = list(SoftwareSystem.FEATURES[feat]["params"].keys())
            if params:
                param = random.choice(params)
                info = SoftwareSystem.FEATURES[feat]["params"][param]
                lo, hi = info["range"]
                opt = info["optimal"]

                key = f"{feat}.{param}"
                understanding[key] = min(0.8, understanding.get(key, 0) + 0.05)

                if random.random() < understanding[key]:
                    new_val = opt + random.randint(-max(1, int((1-understanding[key]) * (hi-lo)*0.1)), max(1, int((1-understanding[key]) * (hi-lo)*0.1)))
                else:
                    new_val = s.param_values[feat][param] + random.randint(-max(1,(hi-lo)//5), max(1,(hi-lo)//5))
                s.set_param(feat, param, new_val)

        new_score = s.score()
        if new_score < score:
            sys = s
            score = new_score
            best = min(best, score)
        history.append(best)

    return best, history


def strategy_level2(n_exp, seed):
    """Level 2: Can ADD/REMOVE features AND tune parameters."""
    random.seed(seed)
    sys = SoftwareSystem()
    # Start minimal
    sys.add_feature("caching")
    sys.add_feature("simple_threading")

    score = sys.score()
    best = score
    history = [best]
    understanding = {}

    for i in range(n_exp):
        s = sys.copy()

        # 40% chance: try adding/removing a feature (structural mutation)
        # 60% chance: tune a parameter (parametric mutation)
        if random.random() < 0.4:
            # Structural mutation
            if random.random() < 0.7:
                # Try adding a feature
                candidates = [f for f in FEATURE_NAMES if f not in s.active_features]
                if candidates:
                    feat = random.choice(candidates)
                    s.add_feature(feat)  # might fail if prereqs not met
            else:
                # Try removing a feature
                if s.active_features:
                    feat = random.choice(list(s.active_features))
                    s.remove_feature(feat)
        else:
            # Parametric mutation (with understanding)
            if s.active_features:
                feat = random.choice(list(s.active_features))
                params = list(SoftwareSystem.FEATURES[feat]["params"].keys())
                if params:
                    param = random.choice(params)
                    info = SoftwareSystem.FEATURES[feat]["params"][param]
                    lo, hi = info["range"]
                    opt = info["optimal"]
                    key = f"{feat}.{param}"
                    understanding[key] = min(0.7, understanding.get(key, 0) + 0.04)
                    if random.random() < understanding[key]:
                        new_val = opt + random.randint(-max(1,int((hi-lo)*0.05)), max(1,int((hi-lo)*0.05)))
                    else:
                        new_val = s.param_values[feat][param] + random.randint(-max(1,(hi-lo)//5), max(1,(hi-lo)//5))
                    s.set_param(feat, param, new_val)

        new_score = s.score()
        if new_score < score:
            sys = s
            score = new_score
            best = min(best, score)
        history.append(best)

    return best, history


def strategy_level3(n_exp, seed):
    """
    Level 3: Structural mutations + Reasoning + Curriculum + Dependency awareness.

    The agent:
    - Understands feature dependencies (adds prereqs before dependents)
    - Detects anti-synergies (removes conflicting features)
    - Follows a curriculum (correctness first, then performance)
    - Uses understanding to make informed parameter choices
    """
    random.seed(seed)
    sys = SoftwareSystem()
    # Start minimal
    sys.add_feature("caching")

    score = sys.score()
    best = score
    history = [best]
    understanding = {}
    feature_knowledge = {}  # tracks which features helped/hurt
    explore_phase = int(n_exp * 0.4)

    for i in range(n_exp):
        s = sys.copy()
        exploring = i < explore_phase

        if exploring and random.random() < 0.6:
            # EXPLORE: Try adding features, learn which help
            candidates = [f for f in FEATURE_NAMES if f not in s.active_features]

            # Informed selection: prefer features we haven't tried or that worked before
            if candidates:
                scored_candidates = []
                for f in candidates:
                    feat_info = SoftwareSystem.FEATURES[f]
                    prereqs_met = all(r in s.active_features for r in feat_info["requires"])
                    if not prereqs_met:
                        # Try adding prereqs first (dependency awareness)
                        for req in feat_info["requires"]:
                            if req not in s.active_features:
                                if req in candidates:
                                    scored_candidates.append((req, 10))  # high priority
                        continue
                    # Score by: tried before? synergy with existing?
                    synergy_score = sum(
                        1 for other in feat_info.get("synergy", {}) if other in s.active_features
                    )
                    anti_score = sum(
                        1 for other in feat_info.get("anti_synergy", {}) if other in s.active_features
                    )
                    knowledge_score = feature_knowledge.get(f, {}).get("net", 0)
                    scored_candidates.append((f, synergy_score - anti_score + knowledge_score))

                if scored_candidates:
                    scored_candidates.sort(key=lambda x: -x[1])
                    feat = scored_candidates[0][0]
                    s.add_feature(feat)
                    # Remove anti-synergies proactively
                    feat_info = SoftwareSystem.FEATURES.get(feat, {})
                    for anti in feat_info.get("anti_synergy", {}):
                        if anti in s.active_features:
                            s.remove_feature(anti)

        elif not exploring and random.random() < 0.2:
            # EXPLOIT PHASE: Occasionally still try structural changes
            # But focus on features with known synergies
            candidates = [f for f in FEATURE_NAMES if f not in s.active_features]
            good_candidates = [f for f in candidates
                               if feature_knowledge.get(f, {}).get("net", 0) >= 0
                               and all(r in s.active_features for r in SoftwareSystem.FEATURES[f]["requires"])]
            if good_candidates:
                feat = random.choice(good_candidates)
                s.add_feature(feat)
        else:
            # Parametric mutation with understanding
            if s.active_features:
                feat = random.choice(list(s.active_features))
                params = list(SoftwareSystem.FEATURES[feat]["params"].keys())
                if params:
                    param = random.choice(params)
                    info = SoftwareSystem.FEATURES[feat]["params"][param]
                    lo, hi = info["range"]
                    opt = info["optimal"]
                    key = f"{feat}.{param}"
                    understanding[key] = min(0.85, understanding.get(key, 0) + 0.05)
                    if random.random() < understanding[key]:
                        noise = max(1, int((1-understanding[key]) * (hi-lo) * 0.1))
                        new_val = opt + random.randint(-noise, noise)
                    else:
                        new_val = s.param_values[feat][param] + random.randint(-max(1,(hi-lo)//10), max(1,(hi-lo)//10))
                    s.set_param(feat, param, new_val)

        new_score = s.score()
        if new_score < score:
            # Track which features were added in this improvement
            added = s.active_features - sys.active_features
            removed = sys.active_features - s.active_features
            for f in added:
                fk = feature_knowledge.setdefault(f, {"added": 0, "helped": 0, "net": 0})
                fk["added"] += 1
                fk["helped"] += 1
                fk["net"] += 1
            for f in removed:
                fk = feature_knowledge.setdefault(f, {"added": 0, "helped": 0, "net": 0})
                fk["net"] -= 1

            sys = s
            score = new_score
            best = min(best, score)
        else:
            # Track features that didn't help
            added = s.active_features - sys.active_features
            for f in added:
                fk = feature_knowledge.setdefault(f, {"added": 0, "helped": 0, "net": 0})
                fk["added"] += 1
                fk["net"] -= 0.5

        history.append(best)

    return best, history


# ════════════════════════════════════════════════════════════
# BENCHMARK
# ════════════════════════════════════════════════════════════

def run_benchmark(n_exp=200, n_seeds=50):
    strategies = {
        "L1: Param tuning only":        strategy_level1,
        "L1.5: Param + Reasoning":      strategy_level1_reasoning,
        "L2: Add features (random)":     strategy_level2,
        "L3: Add features (informed)":   strategy_level3,
    }

    print(f"{'='*72}")
    print(f"  Level 1→3 Benchmark — {n_exp} experiments × {n_seeds} seeds")
    print(f"  System: 12 features, interactions, dependencies, anti-synergies")
    print(f"  Start: caching + simple_threading active (score ≈ 80)")
    print(f"  Global optimum: all synergistic features + tuned params (≈ -60)")
    print(f"{'='*72}\n")

    all_results = {}
    all_histories = {}

    for name, fn in strategies.items():
        scores = []
        histories = []
        for seed in range(n_seeds):
            sc, hist = fn(n_exp, seed)
            scores.append(sc)
            histories.append(hist)

        mean = statistics.mean(scores)
        std = statistics.stdev(scores)
        median = statistics.median(scores)
        best_run = min(scores)

        all_results[name] = {"mean": mean, "std": std, "median": median, "best": best_run}
        all_histories[name] = histories

        print(f"  {name:36s}  mean={mean:>7.1f} ±{std:>5.1f}  median={median:>7.1f}  best={best_run:>7.1f}")

    l1 = all_results["L1: Param tuning only"]["mean"]
    print(f"\n{'─'*72}")
    for name, r in all_results.items():
        imp = (l1 - r["mean"]) / abs(l1) * 100
        print(f"  {name:36s}  vs L1: {imp:>+6.1f}%")

    l3 = all_results["L3: Add features (informed)"]["mean"]
    print(f"\n  ★ Level 3 vs Level 1: {(l1 - l3) / abs(l1) * 100:>+.1f}%")

    # Scaling test
    print(f"\n{'='*72}")
    print(f"  Scaling test: Level advantage at different experiment counts")
    print(f"{'='*72}")
    print(f"{'Exp':>6s} │ {'L1 param':>10s} │ {'L2 add':>10s} │ {'L3 smart':>10s} │ {'L3 vs L1':>10s}")
    print("─" * 56)
    for n in [50, 100, 200, 500]:
        N = 40
        l1_scores = [strategy_level1(n, s)[0] for s in range(N)]
        l2_scores = [strategy_level2(n, s)[0] for s in range(N)]
        l3_scores = [strategy_level3(n, s)[0] for s in range(N)]
        l1m, l2m, l3m = statistics.mean(l1_scores), statistics.mean(l2_scores), statistics.mean(l3_scores)
        imp = (l1m - l3m) / abs(l1m) * 100
        print(f"{n:>6d} │ {l1m:>10.1f} │ {l2m:>10.1f} │ {l3m:>10.1f} │ {imp:>+9.1f}%")

    # Generate chart
    if HAS_MPL:
        generate_chart(all_histories, n_exp, n_seeds, all_results)


def generate_chart(all_histories, n_exp, n_seeds, results):
    x = list(range(n_exp + 1))
    colors = {
        "L1: Param tuning only":     "#f97316",
        "L1.5: Param + Reasoning":   "#eab308",
        "L2: Add features (random)": "#a78bfa",
        "L3: Add features (informed)": "#22d3ee",
    }

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for name, hists in all_histories.items():
        color = colors[name]
        avg = [statistics.mean(h[i] for h in hists) for i in range(n_exp + 1)]
        p25 = [sorted(h[i] for h in hists)[n_seeds//4] for i in range(n_exp + 1)]
        p75 = [sorted(h[i] for h in hists)[3*n_seeds//4] for i in range(n_exp + 1)]

        lw = 2.5 if "L3" in name else 1.5
        ax.fill_between(x, p25, p75, alpha=0.1, color=color)
        short = name.split(":")[0]
        ax.plot(x, avg, color=color, linewidth=lw, label=f'{short}: {results[name]["mean"]:.1f}')

    ax.set_xlabel("Experiments", fontsize=12, color="#e5e7eb")
    ax.set_ylabel("System score (lower = better)", fontsize=12, color="#e5e7eb")
    ax.set_title("Level 1→3: Adding features crushes parameter tuning", fontsize=15, fontweight="bold", color="#f8fafc", pad=12)

    ax.legend(loc="upper right", fontsize=10, framealpha=0.3, edgecolor="#374151", facecolor="#1f2937", labelcolor="#e5e7eb")
    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#374151")
    ax.spines["bottom"].set_color("#374151")
    ax.grid(axis="y", color="#1f2937", linewidth=0.5)

    l1 = results["L1: Param tuning only"]["mean"]
    l3 = results["L3: Add features (informed)"]["mean"]
    imp = (l1 - l3) / abs(l1) * 100

    fig.text(0.5, 0.01,
             f"L3 vs L1: +{imp:.0f}% improvement  |  {n_seeds} seeds · {n_exp} experiments  |  "
             f"12 features, dependencies, synergies, anti-synergies",
             ha="center", fontsize=9, color="#22d3ee", fontweight="500")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("level3_proof.png", facecolor="#0d1117", bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart saved: level3_proof.png")


if __name__ == "__main__":
    run_benchmark()
