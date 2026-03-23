"""
DeepResearch Pipeline — Unified L1-L3 Experiment Runner

Bridges the Level 1 strategy engine (bandit, annealing, population)
with the Level 2-3 engine (mutations, curriculum, orchestrator).

This is what the LLM agent calls to run ONE experiment at any level.
It handles all the wiring: check curriculum stage → pick mutation type →
apply safety rails → run strategy engine → log everything.

Usage:
    from engine.pipeline import ExperimentPipeline
    
    pipe = ExperimentPipeline(project_root=".")
    
    # The agent calls this for every experiment
    instruction = pipe.next_experiment()
    # → tells the agent what to do, which mutation type, which files
    
    # After the agent makes changes:
    result = pipe.evaluate_and_decide()
    # → runs tests, scores, keeps/reverts, updates all state
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from engine.mutations import MutationManager, MutationProposal, MUTATION_TYPES
from engine.curriculum import CurriculumRunner

DR_DIR = Path(".deepresearch")


class ExperimentPipeline:
    """
    Unified experiment runner for all levels.
    
    Per experiment:
    1. Check curriculum stage → get recommended mutation strategy
    2. Consult bandit → which category?
    3. Agent: Deep Read → Hypothesis → Choose mutation type + target
    4. Safety check → snapshot → pre-test
    5. Agent: Apply mutation
    6. Post-test → score → keep/revert
    7. Reflection → log → update bandit + curriculum + knowledge
    """

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.mutations = MutationManager(project_root)
        self.curriculum = CurriculumRunner()
        self.config = self._load_config()
        self.strategy_state = self._load_strategy()
        self.experiment_count = self.strategy_state.get("total_experiments", 0)

    def _load_config(self) -> dict:
        p = DR_DIR / "config.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}

    def _load_strategy(self) -> dict:
        p = DR_DIR / "strategy-state.json"
        if p.exists():
            return json.loads(p.read_text())
        return {"temperature": 1.0, "total_experiments": 0, "bandit_arms": {},
                "no_improvement_streak": 0, "best_metric": None, "baseline_metric": None}

    def _save_strategy(self):
        p = DR_DIR / "strategy-state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.strategy_state, indent=2))

    @property
    def temperature(self) -> float:
        return self.strategy_state.get("temperature", 1.0)

    def next_experiment(self) -> dict:
        """
        Generate instructions for the next experiment.
        
        Returns a dict the LLM agent uses to decide what to do:
        {
            "experiment_id": "exp-0042",
            "curriculum_stage": {"name": ..., "metric": ..., "target": ...},
            "recommended_mutation_types": ["structural_addition", "parametric"],
            "recommended_focus": ["caching", "error_handling"],
            "temperature": 0.6,
            "phase": "explore" or "exploit",
            "instructions": "human-readable guidance for the agent"
        }
        """
        self.experiment_count += 1

        # Curriculum guidance
        stage = self.curriculum.current_stage()
        strategy = self.curriculum.get_mutation_strategy()

        # Temperature and phase
        total_budget = self.config.get("experiment_budget", 200)
        progress = self.experiment_count / max(total_budget, 1)
        phase = "explore" if progress < 0.35 else "exploit"

        # Temperature decay with reheat
        streak = self.strategy_state.get("no_improvement_streak", 0)
        if streak >= 8:
            t = min(0.7, self.temperature + 0.2)  # reheat
            self.strategy_state["no_improvement_streak"] = 0
        else:
            t = max(0.1, 1.0 - progress * 1.5)  # decay
        self.strategy_state["temperature"] = t

        # Build instructions
        if stage:
            stage_info = f"Stage: {stage['name']} — {stage.get('description', '')}\n"
            stage_info += f"Metric: {stage['metric']} {'≥' if stage.get('direction')=='higher' else '≤'} {stage['target']}"
        else:
            stage_info = "All curriculum stages complete — fine-tuning mode"

        mut_types = strategy.get("preferred_types", ["parametric"])
        focus = strategy.get("focus_areas", [])

        instructions = f"""Experiment #{self.experiment_count} | T={t:.2f} | Phase: {phase}

{stage_info}

REASONING PROTOCOL:
1. DEEP READ the target files. What is the current bottleneck?
2. HYPOTHESIZE: What specific change would improve {stage['metric'] if stage else 'the metric'}?
   Preferred mutation types: {', '.join(mut_types)}
   {"Focus areas: " + ', '.join(focus) if focus else "No specific focus — use your judgment"}
3. PREDICT: How much improvement do you expect? Why?
4. IMPLEMENT: Make ONE focused change.
5. After evaluation, REFLECT: Was your prediction right? Why or why not?

Temperature {t:.2f} → {"bold changes, try new approaches" if t > 0.5 else "careful tuning, small improvements"}
"""

        return {
            "experiment_id": f"exp-{self.experiment_count:04d}",
            "curriculum_stage": stage,
            "recommended_mutation_types": mut_types,
            "recommended_focus": focus,
            "temperature": t,
            "phase": phase,
            "instructions": instructions,
        }

    def evaluate_and_decide(self, proposal: MutationProposal,
                            post_metrics: dict = None) -> dict:
        """
        After the agent applied a mutation, evaluate and decide keep/revert.
        
        Args:
            proposal: The mutation that was applied
            post_metrics: Dict of current metrics after mutation
            
        Returns:
            {
                "status": "kept" / "reverted" / "crashed",
                "improvement": float,
                "curriculum_advanced": bool,
                "regressions": list,
                "reflection_prompt": str,
            }
        """
        result = {
            "status": "kept",
            "improvement": 0.0,
            "curriculum_advanced": False,
            "regressions": [],
            "reflection_prompt": "",
        }

        # Run tests if required
        mt = MUTATION_TYPES.get(proposal.mutation_type, {})
        if mt.get("requires_tests"):
            test_passed, test_output = self.mutations.run_tests()
            if not test_passed:
                result["status"] = "reverted"
                result["reflection_prompt"] = (
                    f"Mutation BROKE TESTS. Output:\n{test_output[:300]}\n\n"
                    "Reflect: What assumption was wrong? What would you do differently?"
                )
                # Update bandit: failure
                self._update_bandit(proposal.mutation_type, success=False)
                self._save_strategy()
                return result

        # Check metric improvement
        if post_metrics:
            self.curriculum.update_metrics(post_metrics)

            # Check for regressions on completed stages
            regressions = self.curriculum.check_regression(post_metrics)
            if regressions:
                result["regressions"] = regressions
                result["status"] = "reverted"
                result["reflection_prompt"] = (
                    f"Mutation caused REGRESSIONS on completed stages: {regressions}\n"
                    "These stages must not regress. Reflect: why did this help the "
                    "current stage but hurt completed ones?"
                )
                self._update_bandit(proposal.mutation_type, success=False)
                self._save_strategy()
                return result

            # Check curriculum advancement
            advanced = self.curriculum.check_advancement()
            result["curriculum_advanced"] = advanced

            # Calculate improvement on current metric
            stage = self.curriculum.current_stage()
            if stage:
                metric_name = stage["metric"]
                current = post_metrics.get(metric_name)
                best = self.strategy_state.get("best_metric")
                direction = stage.get("direction", "higher")

                if current is not None:
                    if best is None:
                        self.strategy_state["best_metric"] = current
                        self.strategy_state["baseline_metric"] = current
                        result["improvement"] = 0.0
                    else:
                        if direction == "higher":
                            improved = current > best
                            result["improvement"] = ((current - best) / abs(best) * 100) if best != 0 else 0
                        else:
                            improved = current < best
                            result["improvement"] = ((best - current) / abs(best) * 100) if best != 0 else 0

                        if improved:
                            self.strategy_state["best_metric"] = current
                            self.strategy_state["no_improvement_streak"] = 0
                            result["status"] = "kept"
                        else:
                            self.strategy_state["no_improvement_streak"] = \
                                self.strategy_state.get("no_improvement_streak", 0) + 1
                            # Annealing: accept worse with probability based on temperature
                            if self.temperature > 0.3 and abs(result["improvement"]) < 3:
                                result["status"] = "kept"  # accept small regression
                            else:
                                result["status"] = "reverted"

        # Update bandit
        success = result["status"] == "kept" and result["improvement"] > 0
        self._update_bandit(proposal.mutation_type, success=success)

        # Build reflection prompt
        if not result["reflection_prompt"]:
            if result["status"] == "kept":
                result["reflection_prompt"] = (
                    f"Mutation KEPT. Improvement: {result['improvement']:+.2f}%\n"
                    f"{'🎉 Curriculum advanced!' if result['curriculum_advanced'] else ''}\n"
                    "Reflect: Why did this work? What's the next logical experiment?"
                )
            else:
                result["reflection_prompt"] = (
                    f"Mutation REVERTED. Change: {result['improvement']:+.2f}%\n"
                    "Reflect: Why didn't this improve? Was the hypothesis wrong, "
                    "or was the implementation off?"
                )

        # Save state
        self.strategy_state["total_experiments"] = self.experiment_count
        self._save_strategy()

        # Log
        self._log_experiment(proposal, result)

        return result

    def _update_bandit(self, mutation_type: str, success: bool):
        """Update Thompson sampling arms."""
        arms = self.strategy_state.setdefault("bandit_arms", {})
        arm = arms.setdefault(mutation_type, {"alpha": 1, "beta": 1, "trials": 0})
        arm["trials"] += 1
        if success:
            arm["alpha"] += 1
        else:
            arm["beta"] += 1

    def _log_experiment(self, proposal: MutationProposal, result: dict):
        """Append to experiments.jsonl."""
        log_path = DR_DIR / "experiments.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "id": proposal.id,
            "timestamp": datetime.now().isoformat(),
            "experiment_number": self.experiment_count,
            "mutation_type": proposal.mutation_type,
            "target_files": proposal.target_files,
            "description": proposal.description,
            "hypothesis": proposal.hypothesis,
            "confidence": proposal.confidence,
            "status": result["status"],
            "improvement_pct": result["improvement"],
            "curriculum_advanced": result["curriculum_advanced"],
            "temperature": self.temperature,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def status(self) -> str:
        """Full pipeline status."""
        lines = [
            f"{'═'*60}",
            f"  DeepResearch Pipeline Status",
            f"{'═'*60}",
            f"  Experiments: {self.experiment_count}",
            f"  Temperature: {self.temperature:.2f}",
            f"  Best metric: {self.strategy_state.get('best_metric', 'N/A')}",
            f"  No-improvement streak: {self.strategy_state.get('no_improvement_streak', 0)}",
            "",
        ]

        # Bandit arms
        arms = self.strategy_state.get("bandit_arms", {})
        if arms:
            lines.append("  Bandit Arms:")
            for name, arm in sorted(arms.items(), key=lambda x: -x[1].get("alpha", 1)):
                trials = arm.get("trials", 0)
                successes = arm.get("alpha", 1) - 1
                rate = successes / trials * 100 if trials > 0 else 0
                lines.append(f"    {name:30s} {successes}/{trials} ({rate:.0f}%)")
            lines.append("")

        # Curriculum
        if self.curriculum.stages:
            lines.append(self.curriculum.progress_report())

        lines.append(f"{'═'*60}")
        return "\n".join(lines)

    def should_write_memo(self) -> bool:
        """Check if it's time to write a research memo (every 10 experiments)."""
        return self.experiment_count > 0 and self.experiment_count % 10 == 0

    def should_generate_report(self) -> bool:
        """Check if the session should end with a report."""
        budget = self.config.get("experiment_budget", 200)
        return self.experiment_count >= budget
