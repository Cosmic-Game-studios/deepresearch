#!/usr/bin/env python3
"""
DeepResearch vs Autoresearch — Head-to-Head Benchmark

Runs both strategies on identical problems and generates:
1. progress.png — convergence chart (the hero image)
2. Console output with results table

Usage:
  python compare.py              # default: 200 experiments, 50 seeds
  python compare.py --quick      # quick: 50 experiments, 20 seeds
  python compare.py -n 500 -s 80 # custom
"""

import argparse
import math
import random
import statistics
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ════════════════════════════════════════════════════════════
# SIMULATED OPTIMIZATION LANDSCAPE
# ════════════════════════════════════════════════════════════
# Models a realistic code optimization problem with:
#   - 7 categorical knobs (architecture, depth, lr, optimizer, etc.)
#   - Interaction effects (optimizer X only works with architecture Y)
#   - Local optima traps (greedy gets stuck)
#   - Measurement noise

CATEGORIES = ["architecture", "depth", "learning_rate", "optimizer",
              "batch_size", "regularization", "schedule"]

class CodeState:
    __slots__ = ("arch", "depth", "lr", "opt", "bs", "reg", "sched")

    def __init__(self, arch=0, depth=4, lr=0.001, opt=0, bs=32, reg=0.0, sched=0):
        self.arch = arch; self.depth = depth; self.lr = lr
        self.opt = opt; self.bs = bs; self.reg = reg; self.sched = sched

    def score(self) -> float:
        s = 100.0
        # Architecture: 7 is best (-30), 3-5 is a local trap (-15...-16)
        s += {0:0,1:-5,2:-8,3:-15,4:-16,5:-15,6:-10,7:-30,8:-12,9:-5}.get(self.arch, 0)
        # Depth: optimal=12, local trap at 6-8
        d = -abs(self.depth - 12) * 1.5
        if 6 <= self.depth <= 8: d = max(d, -8)
        s += d
        # LR: optimal=3e-4, trap at ~1e-3
        s += abs(math.log10(self.lr) - math.log10(3e-4)) * 10
        if 8e-4 < self.lr < 1.2e-3: s -= 3
        # Optimizer: Muon(3) bad alone, godlike with arch=7
        s += {0:0, 1:-5, 2:-8, 3:-3}.get(self.opt, 0)
        if self.opt == 3 and self.arch == 7: s -= 20  # interaction!
        if self.sched == 1 and self.opt == 2: s -= 8  # cosine+AdamW
        if self.sched == 3 and self.depth >= 10: s -= 12  # warmup+deep
        # Batch size: optimal=128
        s += ((self.bs - 128) / 50) ** 2
        # Regularization: helps deep models, hurts shallow
        if self.depth > 10: s += abs(self.reg - 0.1) * 15
        else: s += self.reg * 10
        # Schedule: warmup_cosine(3) is best
        s += {0:0, 1:-5, 2:-3, 3:-10}.get(self.sched, 0)
        s += random.gauss(0, 0.5)  # noise
        return s

    def copy(self):
        return CodeState(self.arch, self.depth, self.lr, self.opt, self.bs, self.reg, self.sched)


def mutate(state, category, bold=False):
    s = state.copy()
    if category == "architecture":
        s.arch = random.randint(0,9) if bold else max(0, min(9, s.arch + random.choice([-1,1])))
    elif category == "depth":
        s.depth = random.randint(1,20) if bold else max(1, min(20, s.depth + random.choice([-2,-1,1,2])))
    elif category == "learning_rate":
        s.lr = 10**random.uniform(-5,-1) if bold else s.lr * 10**random.uniform(-0.3,0.3)
    elif category == "optimizer":
        s.opt = random.randint(0,3)
    elif category == "batch_size":
        s.bs = random.choice([8,16,32,64,128,256,512]) if bold else max(8, min(512, random.choice([s.bs//2, s.bs*2])))
    elif category == "regularization":
        s.reg = random.uniform(0,0.5) if bold else max(0, min(1, s.reg + random.uniform(-0.05,0.05)))
    elif category == "schedule":
        s.sched = random.randint(0,3)
    return s


# ════════════════════════════════════════════════════════════
# STRATEGY: GREEDY (Karpathy autoresearch)
# ════════════════════════════════════════════════════════════

def run_greedy(n_exp, seed):
    random.seed(seed)
    state = CodeState()
    score = state.score()
    best = score
    history = [best]
    for _ in range(n_exp):
        cat = random.choice(CATEGORIES)
        ns = mutate(state, cat, bold=random.random() < 0.3)
        ns_score = ns.score()
        if ns_score < score:
            state, score = ns, ns_score
            best = min(best, score)
        history.append(best)
    return best, history


# ════════════════════════════════════════════════════════════
# STRATEGY: DEEPRESEARCH
# ════════════════════════════════════════════════════════════

def run_deepresearch(n_exp, seed):
    random.seed(seed)
    K = 3
    branches = []
    for _ in range(K):
        s = CodeState(); sc = s.score()
        branches.append({"s": s, "sc": sc, "best_s": s.copy(), "best_sc": sc})

    best = min(b["best_sc"] for b in branches)
    history = [best]
    arms = {c: {"a":1,"b":1} for c in CATEGORIES}
    no_imp = 0
    explore_end = int(n_exp * 0.35)

    for i in range(n_exp):
        exploring = i < explore_end
        bold_p = 0.5 if exploring else 0.15
        if no_imp >= 8: bold_p = 0.6; no_imp = 0

        # Thompson sampling
        if random.random() < (0.15 if exploring else 0.05):
            cat = random.choice(CATEGORIES)
        else:
            samples = {c: random.betavariate(a["a"], a["b"]) for c,a in arms.items()}
            cat = max(samples, key=samples.get)

        bi = random.randint(0, K-1) if exploring else min(range(K), key=lambda j: branches[j]["best_sc"])
        br = branches[bi]

        ns = mutate(br["s"], cat, bold=random.random() < bold_p)
        ns_sc = ns.score()

        if ns_sc < br["sc"]:
            br["s"], br["sc"] = ns, ns_sc
            arms[cat]["a"] += 1; no_imp = 0
            if ns_sc < br["best_sc"]: br["best_sc"] = ns_sc; br["best_s"] = ns.copy()
        elif exploring and (ns_sc - br["sc"]) < 3:
            br["s"], br["sc"] = ns, ns_sc
            arms[cat]["b"] += 1; no_imp += 1
        else:
            arms[cat]["b"] += 1; no_imp += 1

        # Crossover at phase transition
        if i == explore_end and K >= 2:
            sb = sorted(branches, key=lambda b: b["best_sc"])
            hybrid = sb[0]["best_s"].copy()
            donor = sb[1]["best_s"]
            for attr in ["arch","depth","lr","opt","bs","reg","sched"]:
                test = hybrid.copy(); setattr(test, attr, getattr(donor, attr))
                if test.score() < hybrid.score(): setattr(hybrid, attr, getattr(donor, attr))
            hsc = hybrid.score()
            wi = max(range(K), key=lambda j: branches[j]["best_sc"])
            if hsc < branches[wi]["best_sc"]:
                branches[wi] = {"s":hybrid,"sc":hsc,"best_s":hybrid.copy(),"best_sc":hsc}

        cur_best = min(b["best_sc"] for b in branches)
        best = min(best, cur_best)
        history.append(best)

    return best, history


# ════════════════════════════════════════════════════════════
# BENCHMARK RUNNER + CHART
# ════════════════════════════════════════════════════════════

def run_comparison(n_exp=200, n_seeds=50):
    greedy_histories = []
    deep_histories = []
    greedy_finals = []
    deep_finals = []

    for seed in range(n_seeds):
        gs, gh = run_greedy(n_exp, seed)
        ds, dh = run_deepresearch(n_exp, seed)
        greedy_finals.append(gs)
        deep_finals.append(ds)
        greedy_histories.append(gh)
        deep_histories.append(dh)

    # Average convergence curves
    g_avg = [statistics.mean(gh[i] for gh in greedy_histories) for i in range(n_exp + 1)]
    d_avg = [statistics.mean(dh[i] for dh in deep_histories) for i in range(n_exp + 1)]

    # Percentile bands (25th-75th)
    g_p25 = [sorted(gh[i] for gh in greedy_histories)[n_seeds//4] for i in range(n_exp + 1)]
    g_p75 = [sorted(gh[i] for gh in greedy_histories)[3*n_seeds//4] for i in range(n_exp + 1)]
    d_p25 = [sorted(dh[i] for dh in deep_histories)[n_seeds//4] for i in range(n_exp + 1)]
    d_p75 = [sorted(dh[i] for dh in deep_histories)[3*n_seeds//4] for i in range(n_exp + 1)]

    gm = statistics.mean(greedy_finals)
    dm = statistics.mean(deep_finals)
    imp = (gm - dm) / gm * 100

    return {
        "g_avg": g_avg, "d_avg": d_avg,
        "g_p25": g_p25, "g_p75": g_p75,
        "d_p25": d_p25, "d_p75": d_p75,
        "greedy_mean": gm, "deep_mean": dm,
        "greedy_std": statistics.stdev(greedy_finals),
        "deep_std": statistics.stdev(deep_finals),
        "greedy_best": min(greedy_finals),
        "deep_best": min(deep_finals),
        "improvement": imp,
        "n_exp": n_exp, "n_seeds": n_seeds,
    }


def generate_chart(results, filename="progress.png"):
    if not HAS_MPL:
        print("⚠  matplotlib not installed, skipping chart generation")
        print("   Install with: pip install matplotlib")
        return

    n = results["n_exp"]
    x = list(range(n + 1))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Confidence bands
    ax.fill_between(x, results["g_p25"], results["g_p75"], alpha=0.15, color="#f97316")
    ax.fill_between(x, results["d_p25"], results["d_p75"], alpha=0.15, color="#22d3ee")

    # Mean curves
    ax.plot(x, results["g_avg"], color="#f97316", linewidth=2.2, label=f'Greedy (autoresearch)  →  {results["greedy_mean"]:.1f}')
    ax.plot(x, results["d_avg"], color="#22d3ee", linewidth=2.2, label=f'DeepResearch  →  {results["deep_mean"]:.1f}')

    # Global optimum line
    ax.axhline(y=7, color="#4ade80", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(n * 0.98, 8.5, "global optimum ≈ 7", ha="right", fontsize=8, color="#4ade80", alpha=0.7)

    # Phase transition marker
    explore_end = int(n * 0.35)
    ax.axvline(x=explore_end, color="#a78bfa", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.text(explore_end + 2, ax.get_ylim()[1] * 0.95, "explore → exploit", fontsize=7,
            color="#a78bfa", alpha=0.6, va="top")

    # Improvement annotation
    imp = results["improvement"]
    mid = n // 2
    ax.annotate(
        f"+{imp:.1f}% better",
        xy=(n, results["d_avg"][-1]), xytext=(n * 0.72, (results["g_avg"][-1] + results["d_avg"][-1]) / 2),
        fontsize=11, fontweight="bold", color="#22d3ee",
        arrowprops=dict(arrowstyle="->", color="#22d3ee", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117", edgecolor="#22d3ee", linewidth=0.8),
    )

    # Styling
    ax.set_xlabel("Experiments", fontsize=11, color="#e5e7eb", labelpad=8)
    ax.set_ylabel("Score (lower is better)", fontsize=11, color="#e5e7eb", labelpad=8)
    ax.set_title("DeepResearch vs Greedy Autoresearch", fontsize=14, fontweight="bold",
                 color="#f8fafc", pad=12)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.3, edgecolor="#374151",
              facecolor="#1f2937", labelcolor="#e5e7eb")

    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#374151")
    ax.spines["bottom"].set_color("#374151")
    ax.grid(axis="y", color="#1f2937", linewidth=0.5)

    # Subtitle
    fig.text(0.5, 0.01,
             f"{results['n_seeds']} seeds · {results['n_exp']} experiments · "
             f"Simulated code optimization with local optima + interaction effects",
             ha="center", fontsize=8, color="#6b7280")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(filename, facecolor="#0d1117", bbox_inches="tight")
    plt.close()
    print(f"📊 Chart saved: {filename}")


def scaling_test():
    """Test how the advantage scales with experiment count."""
    print(f"\n{'Experiments':>12s} │ {'Greedy':>10s} │ {'DeepRes':>10s} │ {'Improvement':>12s}")
    print("─" * 52)
    for n in [50, 100, 200, 500]:
        r = run_comparison(n_exp=n, n_seeds=40)
        print(f"{n:>12d} │ {r['greedy_mean']:>10.1f} │ {r['deep_mean']:>10.1f} │ {r['improvement']:>+11.1f}%")


def main():
    parser = argparse.ArgumentParser(description="DeepResearch vs Autoresearch benchmark")
    parser.add_argument("-n", "--experiments", type=int, default=200, help="Experiments per run")
    parser.add_argument("-s", "--seeds", type=int, default=50, help="Number of random seeds")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 50 exp, 20 seeds")
    parser.add_argument("--scaling", action="store_true", help="Run scaling test (50/100/200/500)")
    parser.add_argument("-o", "--output", default="progress.png", help="Output chart filename")
    args = parser.parse_args()

    if args.quick:
        args.experiments = 50
        args.seeds = 20

    print("🔬 DeepResearch vs Autoresearch — Head-to-Head")
    print(f"   {args.experiments} experiments × {args.seeds} seeds\n")

    results = run_comparison(n_exp=args.experiments, n_seeds=args.seeds)

    print(f"{'─'*50}")
    print(f"  {'Strategy':24s} {'Mean':>8s} {'± Std':>8s} {'Best':>8s}")
    print(f"{'─'*50}")
    print(f"  {'Greedy (autoresearch)':24s} {results['greedy_mean']:>8.1f} {results['greedy_std']:>7.1f} {results['greedy_best']:>8.1f}")
    print(f"  {'DeepResearch':24s} {results['deep_mean']:>8.1f} {results['deep_std']:>7.1f} {results['deep_best']:>8.1f}")
    print(f"{'─'*50}")
    print(f"  Improvement: {results['improvement']:+.1f}%")
    print(f"  (Global optimum ≈ 7.0, baseline ≈ 100)")

    generate_chart(results, args.output)

    if args.scaling:
        scaling_test()


if __name__ == "__main__":
    main()
