#!/usr/bin/env python3
"""
DeepResearch Level 2-3 Engine — Universal Generative Mutations

Level 1: Change values in existing code        (parametric)
Level 2: Add/remove code blocks                (generative)
Level 3: Build systems from specification      (architectural)

This engine manages the scaffolding for Levels 2-3. It does NOT contain
domain-specific knowledge — that comes from the LLM reading the codebase,
documentation, and forming its own understanding.

The engine provides:
1. Mutation types beyond parametric (structural add/remove/replace)
2. Safety rails (test before/after, hard constraints, rollback)
3. Curriculum management (progressive goals)
4. Domain research protocol (what to read before experimenting)
5. Architecture planning (design before code)

Usage:
    python engine/level3.py plan <spec_file>        # Generate architecture plan
    python engine/level3.py curriculum <config>      # Show curriculum status
    python engine/level3.py mutate-check <files...>  # Verify mutation safety
    python engine/level3.py research <topic>         # Domain research protocol
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

DR_DIR = Path(".deepresearch")


# ════════════════════════════════════════════════════════════
# 1. MUTATION TYPES — Beyond parametric
# ════════════════════════════════════════════════════════════

MUTATION_TYPES = {
    # Level 1 (existing)
    "parametric": {
        "level": 1,
        "description": "Change a value in existing code",
        "examples": ["DEPTH = 8 → DEPTH = 12", "lr = 0.001 → lr = 0.0003"],
        "risk": "low",
        "requires_tests": False,
    },

    # Level 2 (new)
    "structural_addition": {
        "level": 2,
        "description": "Add a new code block, function, class, or module",
        "examples": [
            "Add caching layer to database queries",
            "Add retry logic with exponential backoff",
            "Add input validation to API endpoints",
            "Add connection pooling to HTTP client",
        ],
        "risk": "medium",
        "requires_tests": True,
        "safety": "All existing tests must pass before AND after. New code must have at least one test.",
    },
    "structural_removal": {
        "level": 2,
        "description": "Remove dead code, unnecessary complexity, or redundant logic",
        "examples": [
            "Remove unused caching layer that adds latency",
            "Simplify nested conditionals into lookup table",
            "Replace hand-rolled retry with library",
        ],
        "risk": "medium",
        "requires_tests": True,
        "safety": "All existing tests must pass after removal. Metric must not regress.",
    },
    "structural_replacement": {
        "level": 2,
        "description": "Replace one implementation with a better one",
        "examples": [
            "Replace linear search with hash map",
            "Replace synchronous IO with async",
            "Replace custom parser with battle-tested library",
        ],
        "risk": "high",
        "requires_tests": True,
        "safety": "Old tests pass with new implementation. Behavioral equivalence verified.",
    },
    "integration": {
        "level": 2,
        "description": "Connect two existing components that weren't connected",
        "examples": [
            "Wire monitoring into the request pipeline",
            "Connect cache invalidation to write path",
            "Integrate rate limiter with API gateway",
        ],
        "risk": "medium",
        "requires_tests": True,
        "safety": "Integration tests cover the new connection. No existing behavior changes.",
    },

    # Level 3 (future)
    "architectural": {
        "level": 3,
        "description": "Design and implement a new component from specification",
        "examples": [
            "Design and implement a plugin system",
            "Build a new evaluation module from research",
            "Create a domain-specific optimization pipeline",
        ],
        "risk": "high",
        "requires_tests": True,
        "safety": "Component has its own test suite. Integration tested separately.",
    },
}


# ════════════════════════════════════════════════════════════
# 2. SAFETY RAILS — Test before/after, hard constraints
# ════════════════════════════════════════════════════════════

class SafetyRails:
    """Ensures mutations don't break the system."""

    @staticmethod
    def load_constraints(config_path=None):
        """Load hard constraints from config."""
        config_path = config_path or DR_DIR / "config.json"
        if not config_path.exists():
            return {"test_command": None, "hard_constraints": [], "allowed_files": []}
        config = json.loads(config_path.read_text())
        return {
            "test_command": config.get("test_command"),
            "hard_constraints": config.get("hard_constraints", []),
            "allowed_files": config.get("target_files", []),
            "read_only_files": config.get("read_only_files", []),
        }

    @staticmethod
    def check_mutation_safety(mutation_type: str, changed_files: list, constraints: dict) -> dict:
        """
        Check if a proposed mutation is safe to attempt.
        Returns {"safe": bool, "reasons": [...], "warnings": [...]}
        """
        mt = MUTATION_TYPES.get(mutation_type, {})
        result = {"safe": True, "reasons": [], "warnings": []}

        # Check: files are in allowed scope
        allowed = constraints.get("allowed_files", [])
        if allowed:
            for f in changed_files:
                if f not in allowed and not any(f.startswith(a) for a in allowed):
                    result["safe"] = False
                    result["reasons"].append(f"File '{f}' not in target_files scope")

        # Check: read-only files not modified
        read_only = constraints.get("read_only_files", [])
        for f in changed_files:
            if f in read_only:
                result["safe"] = False
                result["reasons"].append(f"File '{f}' is read-only")

        # Check: Level 2+ mutations need tests
        if mt.get("requires_tests") and not constraints.get("test_command"):
            result["warnings"].append(
                f"Mutation type '{mutation_type}' requires tests but no test_command in config. "
                "Proceed with caution — mutation cannot be verified."
            )

        # Check: high-risk mutations get extra warning
        if mt.get("risk") == "high":
            result["warnings"].append(
                f"High-risk mutation type '{mutation_type}'. Ensure you have a git commit to revert to."
            )

        return result

    @staticmethod
    def generate_pre_post_check(test_command: str) -> str:
        """Generate bash commands for pre/post mutation testing."""
        return f"""
# PRE-MUTATION: Run tests to establish baseline
echo "Running pre-mutation tests..."
{test_command} > /tmp/dr-pre-test.log 2>&1
PRE_EXIT=$?
if [ $PRE_EXIT -ne 0 ]; then
    echo "WARNING: Tests already failing before mutation"
    cat /tmp/dr-pre-test.log | tail -20
fi

# ... (agent applies mutation here) ...

# POST-MUTATION: Verify nothing broke
echo "Running post-mutation tests..."
{test_command} > /tmp/dr-post-test.log 2>&1
POST_EXIT=$?
if [ $POST_EXIT -ne 0 ] && [ $PRE_EXIT -eq 0 ]; then
    echo "REVERT: Mutation broke tests that were passing"
    git checkout -- .
    exit 1
fi
echo "Tests passed. Safe to evaluate metric."
"""


# ════════════════════════════════════════════════════════════
# 3. CURRICULUM — Progressive goals, domain-agnostic
# ════════════════════════════════════════════════════════════

class Curriculum:
    """
    Manages a sequence of progressively harder goals.

    A curriculum is defined in .deepresearch/curriculum.json:
    {
      "stages": [
        {
          "name": "Basic correctness",
          "metric": "test_pass_rate",
          "target": 1.0,
          "direction": "higher",
          "description": "All basic tests must pass",
          "unlocks": ["optimization"]
        },
        {
          "name": "Performance optimization",
          "metric": "benchmark_ms",
          "target": 100,
          "direction": "lower",
          "description": "Response time under 100ms",
          "prerequisite": "Basic correctness"
        }
      ]
    }

    The agent advances to the next stage only when the current target is met.
    Each stage can have its own metric, target, and mutation strategy.
    """

    def __init__(self, path=None):
        self.path = path or DR_DIR / "curriculum.json"
        self.stages = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.stages = data.get("stages", [])

    def current_stage(self, metrics: dict) -> dict:
        """Find the first stage whose target hasn't been met."""
        for stage in self.stages:
            metric_name = stage["metric"]
            target = stage["target"]
            direction = stage.get("direction", "higher")
            current = metrics.get(metric_name)

            if current is None:
                return stage  # Can't evaluate — this is the current stage

            if direction == "higher" and current < target:
                return stage
            elif direction == "lower" and current > target:
                return stage

        # All stages complete
        return {"name": "ALL_COMPLETE", "description": "All curriculum stages passed"}

    def progress_report(self, metrics: dict) -> str:
        """Generate a human-readable progress report."""
        lines = ["Curriculum Progress:"]
        for i, stage in enumerate(self.stages):
            metric_name = stage["metric"]
            target = stage["target"]
            direction = stage.get("direction", "higher")
            current = metrics.get(metric_name, "N/A")

            if current == "N/A":
                status = "⬜"
                pct = "?"
            elif direction == "higher":
                passed = current >= target
                status = "✅" if passed else "🔶"
                pct = f"{current/target*100:.0f}%" if target > 0 else "?"
            else:
                passed = current <= target
                status = "✅" if passed else "🔶"
                pct = f"{target/current*100:.0f}%" if current > 0 else "?"

            lines.append(f"  {status} Stage {i+1}: {stage['name']} — {metric_name}: {current} / {target} ({pct})")

        current = self.current_stage(metrics)
        if current.get("name") == "ALL_COMPLETE":
            lines.append("\n  🎉 All stages complete!")
        else:
            lines.append(f"\n  → Current focus: {current['name']}")
            lines.append(f"    {current.get('description', '')}")

        return "\n".join(lines)

    @staticmethod
    def create_template(domain: str) -> dict:
        """Generate a curriculum template for common domains."""
        templates = {
            "web_api": {
                "stages": [
                    {"name": "Correctness", "metric": "test_pass_rate", "target": 1.0, "direction": "higher",
                     "description": "All endpoint tests pass"},
                    {"name": "Performance", "metric": "p99_latency_ms", "target": 100, "direction": "lower",
                     "description": "p99 latency under 100ms"},
                    {"name": "Load handling", "metric": "max_concurrent_users", "target": 1000, "direction": "higher",
                     "description": "Handle 1000 concurrent users"},
                    {"name": "Security", "metric": "security_score", "target": 0.9, "direction": "higher",
                     "description": "Pass security audit with 90%+ score"},
                ]
            },
            "ml_training": {
                "stages": [
                    {"name": "Training runs", "metric": "trains_without_crash", "target": 1.0, "direction": "higher",
                     "description": "Training completes without errors"},
                    {"name": "Baseline", "metric": "val_loss", "target": 2.0, "direction": "lower",
                     "description": "Validation loss below 2.0"},
                    {"name": "Competitive", "metric": "val_loss", "target": 1.0, "direction": "lower",
                     "description": "Validation loss below 1.0"},
                    {"name": "Efficiency", "metric": "throughput_samples_sec", "target": 1000, "direction": "higher",
                     "description": "Process 1000+ samples per second"},
                ]
            },
            "library": {
                "stages": [
                    {"name": "API works", "metric": "test_pass_rate", "target": 1.0, "direction": "higher",
                     "description": "All public API tests pass"},
                    {"name": "Edge cases", "metric": "edge_case_coverage", "target": 0.9, "direction": "higher",
                     "description": "90%+ edge case coverage"},
                    {"name": "Performance", "metric": "benchmark_ops_sec", "target": 10000, "direction": "higher",
                     "description": "10K+ operations per second"},
                    {"name": "Documentation", "metric": "doc_coverage", "target": 0.95, "direction": "higher",
                     "description": "95%+ public API documented"},
                ]
            },
            "game": {
                "stages": [
                    {"name": "Playable", "metric": "no_crash_rate", "target": 1.0, "direction": "higher",
                     "description": "No crashes in 100 simulated games"},
                    {"name": "Balanced", "metric": "win_rate_variance", "target": 0.05, "direction": "lower",
                     "description": "No strategy has >55% win rate"},
                    {"name": "Engaging", "metric": "avg_game_length", "target": 20, "direction": "higher",
                     "description": "Games last at least 20 turns on average"},
                    {"name": "AI challenge", "metric": "ai_vs_random_winrate", "target": 0.9, "direction": "higher",
                     "description": "AI beats random player 90%+ of the time"},
                ]
            },
            "custom": {
                "stages": [
                    {"name": "Stage 1", "metric": "metric_name", "target": 0, "direction": "higher",
                     "description": "Define your first milestone"},
                ]
            },
        }
        return templates.get(domain, templates["custom"])


# ════════════════════════════════════════════════════════════
# 4. DOMAIN RESEARCH PROTOCOL — What to learn before building
# ════════════════════════════════════════════════════════════

RESEARCH_PROTOCOL = """
## Domain Research Protocol (Level 3, Phase 0)

Before writing ANY code, the agent must research the domain.
This replaces the "just start mutating" approach with informed engineering.

### Step 1: Understand the specification
Read the spec/goal. Identify:
- What is the INPUT? (data format, size, frequency)
- What is the OUTPUT? (format, quality requirements, latency)
- What are the CONSTRAINTS? (memory, time, platform, dependencies)
- What is the METRIC? (how do we measure "good enough"?)

### Step 2: Survey existing solutions
Search for existing implementations of similar systems:
- Open source projects (GitHub search)
- Academic papers (if applicable)
- Blog posts and tutorials
- Stack Overflow discussions

Don't copy — LEARN. Understand the common architecture patterns
and why they're used. Note which techniques are standard vs novel.

### Step 3: Identify the core algorithm/architecture
Every domain has a "standard approach" that gets you 80% of the way:
- Web server → request/response loop + middleware + routing
- ML model → data loading → model → training loop → eval
- CLI tool → argument parsing → core logic → output formatting
- Game engine → game loop → state management → AI → rendering

Identify this standard approach. It's your starting architecture.

### Step 4: Plan the implementation order
Break the system into components. Order them by dependency:
1. Core data structures (always first)
2. Input/output handling (can test early)
3. Core algorithm (the hard part)
4. Optimization (Level 1-2 techniques)
5. Edge cases and error handling (last)

### Step 5: Define the curriculum
Write .deepresearch/curriculum.json with progressive goals.
Each stage should be testable and build on the previous.

### Step 6: Begin the experiment loop
NOW start the DeepResearch loop — but with Level 2 mutations enabled.
Each experiment can add code, not just change parameters.
"""


# ════════════════════════════════════════════════════════════
# 5. ARCHITECTURE PLANNER — Design before code
# ════════════════════════════════════════════════════════════

def generate_plan_template(spec: str) -> str:
    """
    Generate an architecture plan template from a specification.
    The LLM fills this in during the research phase.
    """
    return f"""# Architecture Plan
## Generated: {datetime.now().isoformat()}

## Specification
{spec}

## Core Components
<!-- List the main modules/classes/functions needed -->
1. [Component name] — [what it does] — [estimated complexity]
2. ...

## Data Flow
<!-- How data moves through the system -->
Input → [step 1] → [step 2] → ... → Output

## Key Algorithms
<!-- The core algorithms that make this work -->
- [Algorithm]: [why this one, what alternatives exist]

## Dependencies
<!-- External libraries, data, APIs needed -->
- [dependency]: [purpose]

## Component Dependency Order
<!-- What must be built first -->
1. [Foundation component] (no dependencies)
2. [Next component] (depends on #1)
3. ...

## Test Strategy
<!-- How to verify each component works -->
- [Component]: [test approach]

## Curriculum Stages
<!-- Progressive milestones -->
1. [First milestone]: [metric] [target]
2. ...

## Known Risks
<!-- What could go wrong, what we don't know yet -->
- [Risk]: [mitigation]

## Estimated Experiment Budget
<!-- How many experiments to reach each stage -->
- Stage 1: ~[N] experiments
- Stage 2: ~[N] experiments
- Total: ~[N] experiments
"""


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def cmd_plan(spec_file):
    """Generate an architecture plan template."""
    spec = Path(spec_file).read_text() if os.path.exists(spec_file) else spec_file
    plan = generate_plan_template(spec)
    plan_path = DR_DIR / "architecture_plan.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan)
    print(f"Plan template written to {plan_path}")
    print("The agent should fill in this template during the Research Phase.")
    print(RESEARCH_PROTOCOL)

def cmd_curriculum(config_file=None):
    """Show curriculum status."""
    c = Curriculum()
    if not c.stages:
        print("No curriculum defined. Create .deepresearch/curriculum.json")
        print("\nAvailable templates: web_api, ml_training, library, game, custom")
        print("Generate with: python engine/level3.py curriculum-init <domain>")
        return
    # Try to load current metrics
    metrics = {}
    state_path = DR_DIR / "strategy-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("best_metric") is not None:
            metrics[json.loads((DR_DIR / "config.json").read_text())["metric"]] = state["best_metric"]
    print(c.progress_report(metrics))

def cmd_curriculum_init(domain):
    """Create a curriculum template for a domain."""
    template = Curriculum.create_template(domain)
    path = DR_DIR / "curriculum.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2))
    print(f"Curriculum template for '{domain}' written to {path}")
    print(f"Stages: {len(template['stages'])}")
    for i, s in enumerate(template["stages"], 1):
        print(f"  {i}. {s['name']}: {s['metric']} {'>' if s['direction']=='higher' else '<'} {s['target']}")

def cmd_mutate_check(*files):
    """Check if a mutation on given files is safe."""
    constraints = SafetyRails.load_constraints()
    for mt_name, mt in MUTATION_TYPES.items():
        result = SafetyRails.check_mutation_safety(mt_name, list(files), constraints)
        status = "✅" if result["safe"] else "❌"
        print(f"  {status} {mt_name} (Level {mt['level']})")
        for r in result["reasons"]:
            print(f"     ❌ {r}")
        for w in result["warnings"]:
            print(f"     ⚠️  {w}")

def cmd_mutation_types():
    """List all available mutation types."""
    print("Available mutation types:\n")
    for name, mt in MUTATION_TYPES.items():
        print(f"  Level {mt['level']} | {name}")
        print(f"         {mt['description']}")
        print(f"         Risk: {mt['risk']} | Tests required: {mt['requires_tests']}")
        if mt.get("examples"):
            print(f"         Examples: {mt['examples'][0]}")
        print()

def cmd_research(topic):
    """Print the domain research protocol."""
    print(f"Domain Research Protocol for: {topic}")
    print(RESEARCH_PROTOCOL)
    print(f"After research, write findings to .deepresearch/domain_research.md")
    print(f"Then create architecture plan: python engine/level3.py plan '{topic}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("DeepResearch Level 2-3 Engine")
        print()
        print("Commands:")
        print("  mutation-types                    List all mutation types (Level 1-3)")
        print("  mutate-check <file1> [file2...]   Check if mutation is safe")
        print("  plan <spec_or_file>               Generate architecture plan template")
        print("  curriculum                        Show curriculum progress")
        print("  curriculum-init <domain>          Create curriculum (web_api|ml_training|library|game|custom)")
        print("  research <topic>                  Show domain research protocol")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "mutation-types":
        cmd_mutation_types()
    elif cmd == "mutate-check":
        cmd_mutate_check(*sys.argv[2:])
    elif cmd == "plan":
        cmd_plan(sys.argv[2] if len(sys.argv) > 2 else "No specification provided")
    elif cmd == "curriculum":
        cmd_curriculum()
    elif cmd == "curriculum-init":
        cmd_curriculum_init(sys.argv[2] if len(sys.argv) > 2 else "custom")
    elif cmd == "research":
        cmd_research(sys.argv[2] if len(sys.argv) > 2 else "unspecified domain")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
