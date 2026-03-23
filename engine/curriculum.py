"""
DeepResearch Level 2.5 — Curriculum System

Manages progressive goals that build on each other.
Instead of one flat metric, the agent works through stages:
  Stage 1: Make it work (correctness)
  Stage 2: Make it fast (performance)
  Stage 3: Make it scale (load handling)
  Stage 4: Make it robust (reliability)

Each stage has its own metric, target, and mutation strategy.
The agent advances only when the current stage target is met.

Usage:
    from engine.curriculum import CurriculumRunner
    
    runner = CurriculumRunner()
    stage = runner.current_stage()
    print(f"Current focus: {stage['name']}")
    
    # After an experiment, check if we advanced
    runner.update_metrics({"test_pass_rate": 1.0, "p99_ms": 85})
    if runner.check_advancement():
        print(f"Advanced to: {runner.current_stage()['name']}")
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

DR_DIR = Path(".deepresearch")


class CurriculumRunner:
    """
    Manages curriculum progression through stages.
    
    Curriculum definition (.deepresearch/curriculum.json):
    {
      "stages": [
        {
          "name": "Correctness",
          "metric": "test_pass_rate",
          "target": 1.0,
          "direction": "higher",
          "description": "All tests must pass",
          "mutation_strategy": {
            "preferred_types": ["structural_addition", "structural_replacement"],
            "focus_areas": ["error_handling", "input_validation"],
            "temperature": 0.7
          },
          "hard_constraints": ["no regressions on previous stage metrics"],
          "timeout_experiments": 100
        }
      ],
      "advancement_policy": "strict",
      "allow_regression": false
    }
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DR_DIR / "curriculum.json"
        self.history_path = DR_DIR / "curriculum_history.jsonl"
        self.data = self._load()
        self.current_metrics = {}
        self.stage_index = 0
        self._determine_current_stage()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"stages": [], "advancement_policy": "strict", "allow_regression": False}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    @property
    def stages(self) -> list:
        return self.data.get("stages", [])

    def _determine_current_stage(self):
        """Find the first stage whose target hasn't been met."""
        history = self._load_history()
        for i, stage in enumerate(self.stages):
            if not self._stage_completed(stage, history):
                self.stage_index = i
                return
        self.stage_index = len(self.stages)  # all complete

    def _stage_completed(self, stage: dict, history: list) -> bool:
        """Check if a stage was ever completed."""
        metric_name = stage["metric"]
        target = stage["target"]
        direction = stage.get("direction", "higher")
        for entry in history:
            val = entry.get("metrics", {}).get(metric_name)
            if val is not None:
                if direction == "higher" and val >= target:
                    return True
                elif direction == "lower" and val <= target:
                    return True
        return False

    def _load_history(self) -> list:
        if self.history_path.exists():
            lines = self.history_path.read_text().strip().split("\n")
            return [json.loads(l) for l in lines if l.strip()]
        return []

    def _log_history(self, event: str, metrics: dict, stage_name: str):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "stage": stage_name,
            "metrics": metrics,
        }
        with open(self.history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def current_stage(self) -> Optional[dict]:
        """Get the current stage, or None if all complete."""
        if self.stage_index >= len(self.stages):
            return None
        return self.stages[self.stage_index]

    def is_complete(self) -> bool:
        return self.stage_index >= len(self.stages)

    def update_metrics(self, metrics: dict):
        """Update current metrics from latest evaluation."""
        self.current_metrics.update(metrics)
        self._log_history("metric_update", metrics,
                          self.stages[self.stage_index]["name"] if not self.is_complete() else "ALL_COMPLETE")

    def check_advancement(self) -> bool:
        """
        Check if current stage target is met. If so, advance.
        Returns True if we advanced to a new stage.
        """
        if self.is_complete():
            return False

        stage = self.current_stage()
        metric_name = stage["metric"]
        target = stage["target"]
        direction = stage.get("direction", "higher")
        current = self.current_metrics.get(metric_name)

        if current is None:
            return False

        passed = False
        if direction == "higher" and current >= target:
            passed = True
        elif direction == "lower" and current <= target:
            passed = True

        if passed:
            self._log_history("stage_completed", self.current_metrics, stage["name"])
            self.stage_index += 1
            if not self.is_complete():
                self._log_history("stage_started", self.current_metrics,
                                  self.stages[self.stage_index]["name"])
            return True
        return False

    def check_regression(self, metrics: dict) -> list:
        """
        Check if any COMPLETED stage has regressed.
        Returns list of regressed stage names.
        """
        if self.data.get("allow_regression", False):
            return []

        regressions = []
        for i in range(self.stage_index):
            stage = self.stages[i]
            metric_name = stage["metric"]
            target = stage["target"]
            direction = stage.get("direction", "higher")
            current = metrics.get(metric_name)

            if current is None:
                continue
            if direction == "higher" and current < target:
                regressions.append(stage["name"])
            elif direction == "lower" and current > target:
                regressions.append(stage["name"])

        return regressions

    def get_mutation_strategy(self) -> dict:
        """Get the recommended mutation strategy for the current stage."""
        if self.is_complete():
            return {"preferred_types": ["parametric"], "temperature": 0.2,
                    "focus_areas": [], "note": "All stages complete — fine-tuning only"}

        stage = self.current_stage()
        default = {
            "preferred_types": ["parametric", "structural_addition"],
            "temperature": 0.5,
            "focus_areas": [],
        }
        return stage.get("mutation_strategy", default)

    def progress_report(self) -> str:
        """Generate a human-readable progress report."""
        lines = [f"{'═'*60}", "  Curriculum Progress", f"{'═'*60}"]

        for i, stage in enumerate(self.stages):
            metric_name = stage["metric"]
            target = stage["target"]
            direction = stage.get("direction", "higher")
            current = self.current_metrics.get(metric_name)

            if i < self.stage_index:
                status = "✅"
                note = "COMPLETE"
            elif i == self.stage_index:
                status = "🔶"
                note = "← CURRENT"
            else:
                status = "⬜"
                note = "locked"

            if current is not None:
                op = ">=" if direction == "higher" else "<="
                progress = f"{current:.2f} {op} {target}"
                # Progress bar
                if direction == "higher":
                    pct = min(1.0, current / target) if target > 0 else 0
                else:
                    pct = min(1.0, target / current) if current > 0 else 0
                bar_len = 20
                filled = int(pct * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                progress += f"  [{bar}] {pct*100:.0f}%"
            else:
                progress = f"? / {target}"

            lines.append(f"  {status} Stage {i+1}: {stage['name']}")
            lines.append(f"       {metric_name}: {progress}  {note}")
            if stage.get("description"):
                lines.append(f"       {stage['description']}")
            lines.append("")

        if self.is_complete():
            lines.append("  🎉 ALL STAGES COMPLETE")
        else:
            stage = self.current_stage()
            strategy = self.get_mutation_strategy()
            lines.append(f"  Strategy: types={strategy.get('preferred_types', [])}, "
                         f"T={strategy.get('temperature', 0.5)}")

        lines.append(f"{'═'*60}")
        return "\n".join(lines)

    # ─── Curriculum Templates ──────────────────────────────

    @staticmethod
    def templates() -> dict:
        """All available curriculum templates."""
        return {
            "web_api": {
                "stages": [
                    {"name": "Correctness", "metric": "test_pass_rate", "target": 1.0,
                     "direction": "higher", "description": "All endpoint tests pass",
                     "mutation_strategy": {"preferred_types": ["structural_addition", "structural_replacement"],
                                           "focus_areas": ["error_handling", "input_validation"], "temperature": 0.7}},
                    {"name": "Performance", "metric": "p99_latency_ms", "target": 100,
                     "direction": "lower", "description": "p99 latency under 100ms",
                     "mutation_strategy": {"preferred_types": ["structural_addition", "parametric"],
                                           "focus_areas": ["caching", "connection_pooling", "async"], "temperature": 0.5}},
                    {"name": "Load handling", "metric": "max_concurrent_users", "target": 1000,
                     "direction": "higher", "description": "Handle 1000 concurrent users",
                     "mutation_strategy": {"preferred_types": ["structural_addition", "structural_replacement"],
                                           "focus_areas": ["batching", "async_concurrency", "resource_limits"], "temperature": 0.5}},
                    {"name": "Resilience", "metric": "failure_recovery_rate", "target": 0.95,
                     "direction": "higher", "description": "Recover from 95% of failures automatically",
                     "mutation_strategy": {"preferred_types": ["structural_addition", "integration"],
                                           "focus_areas": ["retry_logic", "circuit_breaker", "health_checks"], "temperature": 0.4}},
                ],
                "advancement_policy": "strict", "allow_regression": False,
            },
            "ml_training": {
                "stages": [
                    {"name": "Training runs", "metric": "runs_without_crash", "target": 1.0,
                     "direction": "higher", "description": "Training completes without errors",
                     "mutation_strategy": {"preferred_types": ["structural_replacement", "parametric"],
                                           "temperature": 0.6}},
                    {"name": "Baseline quality", "metric": "val_loss", "target": 2.0,
                     "direction": "lower", "description": "Validation loss below 2.0",
                     "mutation_strategy": {"preferred_types": ["parametric", "structural_addition"],
                                           "focus_areas": ["architecture", "optimizer"], "temperature": 0.7}},
                    {"name": "Competitive quality", "metric": "val_loss", "target": 1.0,
                     "direction": "lower", "description": "Validation loss below 1.0",
                     "mutation_strategy": {"preferred_types": ["parametric", "structural_addition"],
                                           "focus_areas": ["regularization", "data_augmentation", "scheduling"], "temperature": 0.4}},
                    {"name": "Efficiency", "metric": "throughput_samples_sec", "target": 1000,
                     "direction": "higher", "description": "Process 1000+ samples/sec",
                     "mutation_strategy": {"preferred_types": ["structural_replacement", "parametric"],
                                           "focus_areas": ["mixed_precision", "batching", "data_loading"], "temperature": 0.3}},
                ],
                "advancement_policy": "strict", "allow_regression": False,
            },
            "game": {
                "stages": [
                    {"name": "Playable", "metric": "no_crash_rate", "target": 1.0,
                     "direction": "higher", "description": "Zero crashes in simulated games",
                     "mutation_strategy": {"preferred_types": ["structural_addition"], "temperature": 0.6}},
                    {"name": "Balanced", "metric": "max_strategy_winrate", "target": 0.55,
                     "direction": "lower", "description": "No strategy dominates (max 55% win rate)",
                     "mutation_strategy": {"preferred_types": ["parametric", "structural_replacement"], "temperature": 0.5}},
                    {"name": "Engaging", "metric": "avg_game_length_turns", "target": 15,
                     "direction": "higher", "description": "Games last 15+ turns (not too short)",
                     "mutation_strategy": {"preferred_types": ["parametric", "structural_addition"], "temperature": 0.4}},
                    {"name": "AI quality", "metric": "ai_vs_random_winrate", "target": 0.9,
                     "direction": "higher", "description": "AI beats random player 90%+",
                     "mutation_strategy": {"preferred_types": ["structural_addition", "structural_replacement"],
                                           "focus_areas": ["search", "evaluation", "heuristics"], "temperature": 0.5}},
                ],
                "advancement_policy": "strict", "allow_regression": False,
            },
            "library": {
                "stages": [
                    {"name": "Core API", "metric": "core_test_pass_rate", "target": 1.0,
                     "direction": "higher", "description": "All public API tests pass",
                     "mutation_strategy": {"preferred_types": ["structural_addition"], "temperature": 0.6}},
                    {"name": "Edge cases", "metric": "edge_case_pass_rate", "target": 0.95,
                     "direction": "higher", "description": "95%+ edge case coverage",
                     "mutation_strategy": {"preferred_types": ["structural_addition", "parametric"], "temperature": 0.5}},
                    {"name": "Performance", "metric": "benchmark_ops_sec", "target": 10000,
                     "direction": "higher", "description": "10K+ operations per second",
                     "mutation_strategy": {"preferred_types": ["structural_replacement", "parametric"],
                                           "focus_areas": ["algorithm_choice", "data_structures"], "temperature": 0.4}},
                    {"name": "API polish", "metric": "api_consistency_score", "target": 0.9,
                     "direction": "higher", "description": "Consistent naming, error handling, types",
                     "mutation_strategy": {"preferred_types": ["structural_replacement"], "temperature": 0.3}},
                ],
                "advancement_policy": "strict", "allow_regression": False,
            },
            "optimization": {
                "stages": [
                    {"name": "Working baseline", "metric": "tests_pass", "target": 1.0,
                     "direction": "higher", "description": "Existing tests pass",
                     "mutation_strategy": {"preferred_types": ["parametric"], "temperature": 0.3}},
                    {"name": "Low-hanging fruit", "metric": "primary_metric", "target": 0.8,
                     "direction": "higher", "description": "80% of target via easy wins",
                     "mutation_strategy": {"preferred_types": ["parametric", "structural_addition"], "temperature": 0.6}},
                    {"name": "Diminishing returns", "metric": "primary_metric", "target": 0.95,
                     "direction": "higher", "description": "95% of target via harder changes",
                     "mutation_strategy": {"preferred_types": ["structural_replacement", "parametric"], "temperature": 0.4}},
                    {"name": "Final polish", "metric": "primary_metric", "target": 1.0,
                     "direction": "higher", "description": "Hit the target",
                     "mutation_strategy": {"preferred_types": ["parametric"], "temperature": 0.2}},
                ],
                "advancement_policy": "flexible", "allow_regression": False,
            },
            "custom": {
                "stages": [
                    {"name": "Stage 1", "metric": "your_metric", "target": 0,
                     "direction": "higher", "description": "Define your first milestone"},
                ],
                "advancement_policy": "strict", "allow_regression": False,
            },
        }

    @classmethod
    def create_from_template(cls, domain: str, path: Optional[Path] = None) -> "CurriculumRunner":
        """Create a curriculum from a template."""
        templates = cls.templates()
        if domain not in templates:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(templates.keys())}")
        data = templates[domain]
        p = path or DR_DIR / "curriculum.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
        return cls(p)
