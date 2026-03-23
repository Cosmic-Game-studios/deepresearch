#!/usr/bin/env python3
"""
DeepResearch Strategy Engine
Implements: Thompson Sampling bandit, simulated annealing, population management.
Used by the agent to make intelligent experiment decisions.

Usage:
    python strategy.py select          # Select next category + branch
    python strategy.py update <result> # Update after experiment
    python strategy.py status          # Print current strategy state
    python strategy.py report          # Generate session report
"""

import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

DR_DIR = Path(".deepresearch")
CONFIG = DR_DIR / "config.json"
STATE = DR_DIR / "strategy-state.json"
EXPERIMENTS = DR_DIR / "experiments.jsonl"
KNOWLEDGE = DR_DIR / "knowledge.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_experiments():
    if not EXPERIMENTS.exists():
        return []
    exps = []
    for line in EXPERIMENTS.read_text().strip().split("\n"):
        if line.strip():
            exps.append(json.loads(line))
    return exps


def append_experiment(exp):
    with open(EXPERIMENTS, "a") as f:
        f.write(json.dumps(exp) + "\n")


# --- Thompson Sampling ---

def thompson_sample(arms: dict) -> str:
    """Sample from each arm's Beta distribution, return the arm with highest sample."""
    best_arm = None
    best_sample = -1
    for name, stats in arms.items():
        sample = random.betavariate(stats["alpha"], stats["beta"])
        if sample > best_sample:
            best_sample = sample
            best_arm = name
    return best_arm


def select_category(state: dict, config: dict) -> tuple[str, str]:
    """Select mutation category using Thompson Sampling + temperature exploration."""
    arms = state["bandit_arms"]
    temp = state["temperature"]
    categories = config["mutation_categories"]

    # Initialize any missing arms
    for cat in categories:
        if cat not in arms:
            arms[cat] = {"alpha": 1, "beta": 1, "trials": 0}

    # Forced exploration with probability proportional to temperature
    if random.random() < temp * 0.5:
        category = random.choice(categories)
        reason = f"forced_exploration (T={temp:.3f})"
    else:
        category = thompson_sample(arms)
        reason = f"thompson_sampling (T={temp:.3f})"

    return category, reason


# --- Temperature Schedule ---

def compute_temperature(schedule: str, n_experiments: int) -> float:
    """Compute temperature based on schedule and experiment count."""
    schedules = {
        "aggressive":    {"t_init": 1.0, "decay": 0.97},
        "moderate":      {"t_init": 0.5, "decay": 0.95},
        "conservative":  {"t_init": 0.2, "decay": 0.93},
    }
    s = schedules.get(schedule, schedules["moderate"])
    return s["t_init"] * (s["decay"] ** n_experiments)


# --- Simulated Annealing ---

def acceptance_probability(delta: float, temperature: float, baseline_range: float) -> float:
    """
    Calculate probability of accepting a worse result.
    delta: how much worse (positive = worse)
    temperature: current temperature
    baseline_range: metric range for normalization
    """
    if delta <= 0:
        return 1.0  # Always accept improvements
    if temperature < 0.001:
        return 0.0  # Greedy when cold
    if baseline_range == 0:
        return 0.0
    normalized_delta = delta / max(baseline_range, 1e-6)
    return math.exp(-normalized_delta / temperature)


# --- Population Management ---

def select_branch(state: dict) -> str:
    """Select which branch to mutate using tournament selection."""
    population = state.get("population", [])
    if not population:
        return "branch-0"

    temp = state["temperature"]

    # With probability T, pick random branch (diversity)
    if random.random() < temp:
        return random.choice(population)["branch"]

    # Tournament: pick 2 random, return the better one
    if len(population) >= 2:
        a, b = random.sample(population, 2)
        direction = state.get("metric_direction", "lower")
        if direction == "lower":
            return a["branch"] if a["metric"] < b["metric"] else b["branch"]
        else:
            return a["branch"] if a["metric"] > b["metric"] else b["branch"]
    return population[0]["branch"]


def should_crossover(state: dict) -> bool:
    """Check if it's time for a crossover attempt."""
    n = state["total_experiments"]
    return n > 0 and n % 10 == 0 and len(state.get("population", [])) >= 2


def should_ablate(state: dict) -> bool:
    """Check if it's time for ablation analysis."""
    n = state["total_experiments"]
    return n > 0 and n % 20 == 0


# --- Knowledge Base ---

def check_anti_patterns(knowledge: dict, category: str, hypothesis: str) -> list[str]:
    """Check if a proposed experiment matches known anti-patterns."""
    warnings = []
    for ap in knowledge.get("anti_patterns", []):
        if ap["category"] == category and ap["confidence"] > 0.8:
            warnings.append(f"⚠ Anti-pattern: {ap['description']} (confidence: {ap['confidence']:.0%})")
    return warnings


def get_domain_insights(knowledge: dict, domain: str) -> list[str]:
    """Get relevant insights for the current domain."""
    return [
        i["insight"]
        for i in knowledge.get("domain_insights", [])
        if i.get("domain") == domain
    ]


def update_knowledge(knowledge: dict, experiment: dict, experiments: list):
    """Update knowledge base with new experiment data."""
    category = experiment["category"]
    status = experiment["status"]

    # Track pattern frequency
    cat_exps = [e for e in experiments if e["category"] == category]
    cat_successes = [e for e in cat_exps if e["status"] == "kept"]
    cat_failures = [e for e in cat_exps if e["status"] == "reverted"]

    # Update anti-patterns (3+ consecutive failures in a category)
    if len(cat_failures) >= 3:
        recent_3 = cat_exps[-3:]
        if all(e["status"] in ("reverted", "crashed") for e in recent_3):
            existing = next(
                (ap for ap in knowledge["anti_patterns"] if ap["category"] == category),
                None,
            )
            if existing:
                existing["evidence_count"] = len(cat_failures)
                existing["confidence"] = min(0.95, len(cat_failures) / (len(cat_exps) + 1))
            else:
                knowledge["anti_patterns"].append({
                    "domain": "auto",
                    "category": category,
                    "description": f"Category '{category}' has {len(cat_failures)} failures in {len(cat_exps)} trials",
                    "confidence": len(cat_failures) / (len(cat_exps) + 1),
                    "evidence_count": len(cat_failures),
                })

    # Track successful patterns
    if status == "kept" and experiment.get("improvement_pct", 0) > 1.0:
        knowledge["patterns"].append({
            "domain": "auto",
            "category": category,
            "description": experiment.get("mutation_description", ""),
            "confidence": 0.6,
            "evidence_count": 1,
            "first_seen": experiment["timestamp"],
            "last_confirmed": experiment["timestamp"],
        })

    return knowledge


# --- Plateau Detection ---

def detect_plateau(experiments: list, window: int = 15) -> bool:
    """Detect if we're stuck in a plateau."""
    if len(experiments) < window:
        return False
    recent = experiments[-window:]
    improvements = sum(1 for e in recent if e["status"] == "kept")
    return improvements == 0


def suggest_reheat(state: dict, experiments: list) -> dict:
    """Suggest temperature reheat if stuck."""
    if detect_plateau(experiments):
        return {
            "action": "reheat",
            "new_temperature": min(state["temperature"] * 3, 0.8),
            "reason": f"Plateau detected: {len(experiments)} experiments, 0 improvements in last 15",
        }
    return {"action": "continue"}


# --- Commands ---

def cmd_select():
    """Select next experiment parameters."""
    config = load_json(CONFIG)
    state = load_json(STATE)
    knowledge = load_json(KNOWLEDGE) if KNOWLEDGE.exists() else {"patterns": [], "anti_patterns": [], "domain_insights": []}
    experiments = load_experiments()

    # Update temperature
    state["temperature"] = compute_temperature(
        config.get("temperature_schedule", "moderate"),
        state["total_experiments"],
    )

    # Check for plateau
    reheat = suggest_reheat(state, experiments)
    if reheat["action"] == "reheat":
        state["temperature"] = reheat["new_temperature"]
        print(f"🔥 REHEAT: {reheat['reason']}")
        print(f"   Temperature raised to {state['temperature']:.3f}")

    # Select category
    category, reason = select_category(state, config)

    # Check anti-patterns
    warnings = check_anti_patterns(knowledge, category, "")
    for w in warnings:
        print(w)
        # Re-select if high-confidence anti-pattern
        category, reason = select_category(state, config)

    # Select branch
    branch = select_branch(state)

    # Check for special actions
    special = None
    if should_crossover(state):
        special = "crossover"
    elif should_ablate(state):
        special = "ablation"

    result = {
        "category": category,
        "branch": branch,
        "reason": reason,
        "temperature": state["temperature"],
        "experiment_number": state["total_experiments"] + 1,
        "special_action": special,
    }

    # Get insights for context
    insights = get_domain_insights(knowledge, "auto")
    if insights:
        result["relevant_insights"] = insights[:3]

    save_json(STATE, state)
    print(json.dumps(result, indent=2))


def cmd_update(result_json: str):
    """Update state after an experiment."""
    result = json.loads(result_json)
    state = load_json(STATE)
    experiments = load_experiments()

    # Update bandit arm
    category = result["category"]
    arms = state["bandit_arms"]
    if category in arms:
        arms[category]["trials"] += 1
        if result["status"] in ("kept", "accepted-worse"):
            arms[category]["alpha"] += 1
        else:
            arms[category]["beta"] += 1

    # Update total count
    state["total_experiments"] += 1

    # Update best metric
    if result["status"] == "kept":
        direction = load_json(CONFIG).get("metric_direction", "lower")
        current_best = state.get("best_metric")
        new_metric = result["metric"]
        if current_best is None:
            state["best_metric"] = new_metric
        elif direction == "lower" and new_metric < current_best:
            state["best_metric"] = new_metric
        elif direction == "higher" and new_metric > current_best:
            state["best_metric"] = new_metric

    # Update knowledge base
    knowledge = load_json(KNOWLEDGE) if KNOWLEDGE.exists() else {"patterns": [], "anti_patterns": [], "domain_insights": []}
    knowledge = update_knowledge(knowledge, result, experiments)
    save_json(KNOWLEDGE, knowledge)

    # Append experiment
    append_experiment(result)

    save_json(STATE, state)

    # Status line
    status_icon = {"kept": "✓", "reverted": "✗", "crashed": "💥", "accepted-worse": "~", "baseline": "◆"}
    icon = status_icon.get(result["status"], "?")
    improvement = result.get("improvement_pct", 0)
    print(f"[#{state['total_experiments']} | {result.get('branch','?')} | {category} | T={state['temperature']:.3f}] "
          f"{result.get('metric', '?')} {icon} {result['status']} ({improvement:+.2f}%)")


def cmd_status():
    """Print current strategy state."""
    state = load_json(STATE)
    config = load_json(CONFIG)
    experiments = load_experiments()

    print(f"=== DeepResearch Status ===")
    print(f"Total experiments: {state['total_experiments']}")
    print(f"Temperature: {state['temperature']:.4f}")
    print(f"Baseline: {state.get('baseline_metric', 'N/A')}")
    print(f"Best: {state.get('best_metric', 'N/A')}")

    if state.get("baseline_metric") and state.get("best_metric"):
        baseline = state["baseline_metric"]
        best = state["best_metric"]
        direction = config.get("metric_direction", "lower")
        if direction == "lower":
            improvement = (baseline - best) / baseline * 100
        else:
            improvement = (best - baseline) / baseline * 100
        print(f"Improvement: {improvement:+.2f}%")

    print(f"\n--- Bandit Arms ---")
    for name, stats in sorted(state.get("bandit_arms", {}).items()):
        rate = (stats["alpha"] - 1) / max(stats["trials"], 1) * 100
        print(f"  {name:20s} | trials={stats['trials']:3d} | "
              f"α={stats['alpha']:3d} β={stats['beta']:3d} | "
              f"success={rate:.0f}%")

    # Recent experiments
    if experiments:
        print(f"\n--- Last 5 Experiments ---")
        for exp in experiments[-5:]:
            icon = {"kept": "✓", "reverted": "✗", "crashed": "💥", "accepted-worse": "~"}.get(exp.get("status"), "?")
            print(f"  #{exp['id']:3d} {icon} {exp.get('category','?'):15s} "
                  f"metric={exp.get('metric','?')} {exp.get('status','?')}")

    # Plateau check
    if detect_plateau(experiments):
        print(f"\n⚠️  PLATEAU DETECTED — consider reheat or restart")


def cmd_report():
    """Generate a session report."""
    state = load_json(STATE)
    config = load_json(CONFIG)
    experiments = load_experiments()

    if not experiments:
        print("No experiments to report on.")
        return

    # Compute stats
    total = len(experiments)
    kept = sum(1 for e in experiments if e.get("status") == "kept")
    reverted = sum(1 for e in experiments if e.get("status") == "reverted")
    crashed = sum(1 for e in experiments if e.get("status") == "crashed")
    baseline = state.get("baseline_metric", experiments[0].get("metric"))
    best = state.get("best_metric", baseline)
    direction = config.get("metric_direction", "lower")

    if baseline and best:
        if direction == "lower":
            improvement = (baseline - best) / baseline * 100
        else:
            improvement = (best - baseline) / baseline * 100
    else:
        improvement = 0

    # Top improvements
    improvements = sorted(
        [e for e in experiments if e.get("status") == "kept" and e.get("improvement_pct", 0) > 0],
        key=lambda e: abs(e.get("improvement_pct", 0)),
        reverse=True,
    )[:10]

    # Arm stats
    arms = state.get("bandit_arms", {})

    # Generate report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# DeepResearch Report
**Date:** {now}
**Experiments:** {total} total ({kept} kept, {reverted} reverted, {crashed} crashed)

## Results
- **Baseline:** {baseline}
- **Best:** {best}
- **Improvement:** {improvement:+.2f}%
- **Success rate:** {kept/max(total,1)*100:.1f}%

## Bandit Arm Performance
| Category | Trials | Successes | Rate |
|---|---|---|---|
"""
    for name, stats in sorted(arms.items()):
        rate = (stats["alpha"] - 1) / max(stats["trials"], 1) * 100
        report += f"| {name} | {stats['trials']} | {stats['alpha']-1} | {rate:.0f}% |\n"

    report += "\n## Top Improvements\n"
    for i, exp in enumerate(improvements, 1):
        report += f"{i}. **#{exp['id']}** ({exp['category']}): {exp.get('mutation_description', exp.get('hypothesis', 'N/A'))} → {exp.get('improvement_pct', 0):+.2f}%\n"

    # Failed approaches
    failed_cats = {}
    for e in experiments:
        if e.get("status") in ("reverted", "crashed"):
            cat = e.get("category", "unknown")
            failed_cats[cat] = failed_cats.get(cat, 0) + 1

    if failed_cats:
        report += "\n## Failed Approaches\n"
        for cat, count in sorted(failed_cats.items(), key=lambda x: -x[1]):
            report += f"- **{cat}**: {count} failed attempts\n"

    # Save report
    report_dir = DR_DIR / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"session-{datetime.now().strftime('%Y%m%d-%H%M')}.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python strategy.py [select|update|status|report]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "select":
        cmd_select()
    elif cmd == "update":
        if len(sys.argv) < 3:
            print("Usage: python strategy.py update '<json>'")
            sys.exit(1)
        cmd_update(sys.argv[2])
    elif cmd == "status":
        cmd_status()
    elif cmd == "report":
        cmd_report()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
