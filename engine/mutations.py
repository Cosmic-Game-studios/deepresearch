"""
DeepResearch Level 2 — Mutation System

Manages all mutation types from parametric (L1) to architectural (L3).
Tracks diffs, manages multi-file changes, enforces safety rails.

Usage:
    from engine.mutations import MutationManager
    mm = MutationManager(project_root=".")
    
    # Propose a mutation
    proposal = mm.propose(
        mutation_type="structural_addition",
        target_files=["src/server.py"],
        description="Add connection pooling to database client",
        hypothesis="DB connections are the bottleneck — pool should reduce p99 by 40%"
    )
    
    # Execute with safety
    result = mm.execute(proposal)  # runs tests before/after, reverts on failure
"""

import json
import subprocess
import hashlib
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

DR_DIR = Path(".deepresearch")

# ════════════════════════════════════════════════════════════
# MUTATION TYPES
# ════════════════════════════════════════════════════════════

MUTATION_TYPES = {
    "parametric": {
        "level": 1,
        "description": "Change a value in existing code",
        "risk": "low",
        "requires_tests": False,
        "max_files": 1,
        "max_lines_changed": 5,
    },
    "structural_addition": {
        "level": 2,
        "description": "Add new code block, function, class, or module",
        "risk": "medium",
        "requires_tests": True,
        "max_files": 5,
        "max_lines_changed": 200,
    },
    "structural_removal": {
        "level": 2,
        "description": "Remove dead code, unnecessary complexity, redundant logic",
        "risk": "medium",
        "requires_tests": True,
        "max_files": 5,
        "max_lines_changed": 100,
    },
    "structural_replacement": {
        "level": 2,
        "description": "Replace one implementation with a better one",
        "risk": "high",
        "requires_tests": True,
        "max_files": 10,
        "max_lines_changed": 300,
    },
    "integration": {
        "level": 2,
        "description": "Connect two existing components that weren't connected",
        "risk": "medium",
        "requires_tests": True,
        "max_files": 5,
        "max_lines_changed": 100,
    },
    "architectural": {
        "level": 3,
        "description": "Design and implement a new component from specification",
        "risk": "high",
        "requires_tests": True,
        "max_files": 20,
        "max_lines_changed": 1000,
    },
}


@dataclass
class MutationProposal:
    """A proposed mutation before execution."""
    id: str
    mutation_type: str
    target_files: list
    description: str
    hypothesis: str
    predicted_impact: str = ""
    confidence: str = "medium"  # low/medium/high
    depends_on: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class MutationResult:
    """Result of an executed mutation."""
    proposal_id: str
    status: str  # "kept", "reverted", "crashed", "blocked"
    pre_test_passed: bool = True
    post_test_passed: bool = True
    pre_metric: Optional[float] = None
    post_metric: Optional[float] = None
    improvement_pct: float = 0.0
    files_changed: list = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    error: str = ""
    duration_seconds: float = 0.0
    reflection: str = ""

    def to_dict(self):
        return asdict(self)


class MutationManager:
    """Orchestrates mutations with safety rails."""

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.dr_dir = self.root / ".deepresearch"
        self.config = self._load_config()
        self.counter = self._get_next_id()

    def _load_config(self) -> dict:
        config_path = self.dr_dir / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {
            "test_command": None,
            "target_files": [],
            "read_only_files": [],
            "mutation_levels": [1, 2],
            "metric": "score",
            "metric_direction": "lower",
            "hard_constraints": [],
        }

    def _get_next_id(self) -> int:
        log_path = self.dr_dir / "experiments.jsonl"
        if log_path.exists():
            lines = [l for l in log_path.read_text().strip().split("\n") if l.strip()]
            return len(lines) + 1
        return 1

    def propose(self, mutation_type: str, target_files: list,
                description: str, hypothesis: str, **kwargs) -> MutationProposal:
        """Create a mutation proposal. Does NOT execute it."""
        mt = MUTATION_TYPES.get(mutation_type)
        if not mt:
            raise ValueError(f"Unknown mutation type: {mutation_type}. "
                             f"Available: {list(MUTATION_TYPES.keys())}")

        # Check level is allowed
        allowed = self.config.get("mutation_levels", [1])
        if mt["level"] not in allowed and mt["level"] > max(allowed):
            raise ValueError(f"Mutation type '{mutation_type}' requires level {mt['level']}, "
                             f"but config allows {allowed}")

        proposal = MutationProposal(
            id=f"exp-{self.counter:04d}",
            mutation_type=mutation_type,
            target_files=target_files,
            description=description,
            hypothesis=hypothesis,
            **kwargs,
        )
        self.counter += 1
        return proposal

    def check_safety(self, proposal: MutationProposal) -> dict:
        """Check if a proposal is safe to execute. Returns {safe, reasons, warnings}."""
        mt = MUTATION_TYPES[proposal.mutation_type]
        result = {"safe": True, "reasons": [], "warnings": []}

        # Check read-only files
        read_only = self.config.get("read_only_files", [])
        for f in proposal.target_files:
            for ro in read_only:
                if f.startswith(ro) or f == ro:
                    result["safe"] = False
                    result["reasons"].append(f"'{f}' is in read-only scope '{ro}'")

        # Check file count limit
        if len(proposal.target_files) > mt["max_files"]:
            result["warnings"].append(
                f"Changing {len(proposal.target_files)} files (limit: {mt['max_files']})")

        # Check tests required
        if mt["requires_tests"] and not self.config.get("test_command"):
            result["warnings"].append(
                f"'{proposal.mutation_type}' requires tests but no test_command configured")

        # Check high risk
        if mt["risk"] == "high":
            result["warnings"].append(
                f"High-risk mutation. Ensure git commit exists to revert to.")

        return result

    def snapshot_files(self, files: list) -> dict:
        """Take a snapshot of file contents for rollback."""
        snapshot = {}
        for f in files:
            p = self.root / f
            if p.exists():
                snapshot[f] = p.read_text()
            else:
                snapshot[f] = None  # file doesn't exist yet
        return snapshot

    def rollback(self, snapshot: dict):
        """Restore files from snapshot."""
        for f, content in snapshot.items():
            p = self.root / f
            if content is None:
                if p.exists():
                    p.unlink()  # remove file that was created
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)

    def run_tests(self) -> tuple:
        """Run test suite. Returns (passed: bool, output: str)."""
        cmd = self.config.get("test_command")
        if not cmd:
            return True, "No test command configured"
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=300, cwd=str(self.root)
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Tests timed out after 300s"
        except Exception as e:
            return False, f"Test error: {e}"

    def run_eval(self) -> Optional[float]:
        """Run evaluation harness and extract metric."""
        eval_cmd = self.config.get("eval_command")
        if not eval_cmd:
            return None
        try:
            result = subprocess.run(
                eval_cmd, shell=True, capture_output=True, text=True,
                timeout=self.config.get("budget_seconds", 300),
                cwd=str(self.root)
            )
            # Try to extract metric from output
            metric_name = self.config.get("metric", "score")
            for line in result.stdout.strip().split("\n"):
                if metric_name in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            return float(p)
                        except ValueError:
                            continue
            return None
        except Exception:
            return None

    def execute(self, proposal: MutationProposal,
                apply_fn=None) -> MutationResult:
        """
        Execute a mutation with full safety rails.
        
        apply_fn: A callable that applies the mutation. If None, the caller
                  must apply the mutation between propose() and execute().
                  This is the normal flow for LLM agents — they write the code,
                  then call execute() to test and score it.
        """
        start = time.time()
        result = MutationResult(proposal_id=proposal.id)

        # 1. Safety check
        safety = self.check_safety(proposal)
        if not safety["safe"]:
            result.status = "blocked"
            result.error = "; ".join(safety["reasons"])
            return result

        # 2. Snapshot for rollback
        snapshot = self.snapshot_files(proposal.target_files)

        # 3. Pre-mutation tests
        mt = MUTATION_TYPES[proposal.mutation_type]
        if mt["requires_tests"]:
            passed, output = self.run_tests()
            result.pre_test_passed = passed
            if not passed:
                result.status = "blocked"
                result.error = f"Pre-mutation tests failing: {output[:500]}"
                return result

        # 4. Pre-mutation metric
        result.pre_metric = self.run_eval()

        # 5. Apply mutation (if apply_fn provided)
        if apply_fn:
            try:
                apply_fn()
            except Exception as e:
                self.rollback(snapshot)
                result.status = "crashed"
                result.error = f"Mutation failed: {e}"
                result.duration_seconds = time.time() - start
                return result

        # 6. Post-mutation tests
        if mt["requires_tests"]:
            passed, output = self.run_tests()
            result.post_test_passed = passed
            if not passed and result.pre_test_passed:
                # Tests broke — REVERT
                self.rollback(snapshot)
                result.status = "reverted"
                result.error = f"Mutation broke tests: {output[:500]}"
                result.duration_seconds = time.time() - start
                return result

        # 7. Post-mutation metric
        result.post_metric = self.run_eval()

        # 8. Compare metrics
        if result.pre_metric is not None and result.post_metric is not None:
            direction = self.config.get("metric_direction", "lower")
            if direction == "lower":
                improved = result.post_metric < result.pre_metric
                if result.pre_metric != 0:
                    result.improvement_pct = (result.pre_metric - result.post_metric) / abs(result.pre_metric) * 100
            else:
                improved = result.post_metric > result.pre_metric
                if result.pre_metric != 0:
                    result.improvement_pct = (result.post_metric - result.pre_metric) / abs(result.pre_metric) * 100

            if improved:
                result.status = "kept"
            else:
                self.rollback(snapshot)
                result.status = "reverted"
        else:
            # No metric available — keep if tests pass
            result.status = "kept" if result.post_test_passed else "reverted"

        # 9. Calculate diff stats
        for f in proposal.target_files:
            p = self.root / f
            if p.exists():
                result.files_changed.append(f)
                old = snapshot.get(f, "")
                new = p.read_text() if p.exists() else ""
                old_lines = (old or "").split("\n")
                new_lines = new.split("\n")
                result.lines_added += max(0, len(new_lines) - len(old_lines))
                result.lines_removed += max(0, len(old_lines) - len(new_lines))

        result.duration_seconds = time.time() - start
        return result

    def log_result(self, proposal: MutationProposal, result: MutationResult):
        """Append result to experiment log."""
        log_path = self.dr_dir / "experiments.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "id": proposal.id,
            "timestamp": datetime.now().isoformat(),
            "mutation_type": proposal.mutation_type,
            "target_files": proposal.target_files,
            "description": proposal.description,
            "hypothesis": proposal.hypothesis,
            "predicted_impact": proposal.predicted_impact,
            "confidence": proposal.confidence,
            "depends_on": proposal.depends_on,
            **result.to_dict(),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ════════════════════════════════════════════════════════════
# FEATURE DISCOVERY — Analyze codebase to find what's missing
# ════════════════════════════════════════════════════════════

class FeatureDiscovery:
    """
    Helps the agent discover what features are MISSING from a codebase.
    
    This is NOT a pre-built feature library. Instead, it provides PATTERNS
    the agent can look for — universal software engineering improvements
    that apply across domains.
    
    The agent uses its Reasoning Layer (R1 Deep Read) to decide which
    of these patterns would actually help for the specific codebase.
    """

    # Universal improvement patterns (domain-agnostic)
    PATTERNS = {
        "performance": [
            {"pattern": "caching", "signal": "Repeated expensive computations or IO calls",
             "check": "Look for functions called multiple times with same inputs"},
            {"pattern": "batching", "signal": "Many small operations that could be grouped",
             "check": "Look for loops that make individual calls (DB, API, IO)"},
            {"pattern": "lazy_loading", "signal": "Loading data that might not be used",
             "check": "Look for initialization that loads everything upfront"},
            {"pattern": "indexing", "signal": "Linear searches through collections",
             "check": "Look for loops that search/filter without index structures"},
            {"pattern": "connection_pooling", "signal": "Creating new connections per request",
             "check": "Look for connection creation inside request handlers"},
            {"pattern": "async_concurrency", "signal": "Sequential IO-bound operations",
             "check": "Look for sequential await/request chains that could be parallel"},
        ],
        "reliability": [
            {"pattern": "error_handling", "signal": "Unhandled exceptions or bare try/except",
             "check": "Look for missing error handling on IO, parsing, external calls"},
            {"pattern": "retry_logic", "signal": "Transient failures crash the system",
             "check": "Look for external calls without retry on timeout/5xx"},
            {"pattern": "circuit_breaker", "signal": "Cascading failures from downstream",
             "check": "Look for external dependencies without failure isolation"},
            {"pattern": "input_validation", "signal": "Trusting external input",
             "check": "Look for user/API input used without validation"},
            {"pattern": "rate_limiting", "signal": "Unbounded resource consumption",
             "check": "Look for endpoints without rate/concurrency limits"},
        ],
        "maintainability": [
            {"pattern": "dead_code_removal", "signal": "Unused imports, functions, variables",
             "check": "Look for code that's never called or always short-circuited"},
            {"pattern": "duplication_removal", "signal": "Copy-pasted logic",
             "check": "Look for similar code blocks that differ in small ways"},
            {"pattern": "abstraction", "signal": "Long functions doing multiple things",
             "check": "Look for functions >50 lines or with multiple responsibilities"},
            {"pattern": "configuration", "signal": "Magic numbers and hardcoded values",
             "check": "Look for hardcoded URLs, thresholds, sizes, timeouts"},
        ],
        "observability": [
            {"pattern": "logging", "signal": "Hard to debug failures",
             "check": "Look for error paths without logging"},
            {"pattern": "metrics", "signal": "No visibility into performance",
             "check": "Look for missing latency/throughput/error rate tracking"},
            {"pattern": "health_checks", "signal": "No way to verify system is running",
             "check": "Look for missing /health or equivalent endpoints"},
        ],
    }

    @classmethod
    def suggest_analysis(cls, category: str = None) -> list:
        """Get analysis prompts for the agent to evaluate."""
        if category:
            return cls.PATTERNS.get(category, [])
        return [p for patterns in cls.PATTERNS.values() for p in patterns]

    @classmethod
    def generate_analysis_prompt(cls) -> str:
        """Generate a prompt for the LLM to analyze a codebase."""
        lines = ["Analyze this codebase for improvement opportunities.\n",
                 "For each pattern below, check if it applies:\n"]
        for cat, patterns in cls.PATTERNS.items():
            lines.append(f"\n### {cat.title()}")
            for p in patterns:
                lines.append(f"- **{p['pattern']}**: {p['check']}")
        lines.append("\nFor each applicable pattern, write a Level 2 mutation proposal:")
        lines.append("- What specifically to add/change/remove")
        lines.append("- Which files to modify")
        lines.append("- Expected impact on the metric")
        lines.append("- What tests to add")
        return "\n".join(lines)
