"""
DeepResearch Level 3 — Autonomous Engineer

The full pipeline: Specification → Research → Architecture → Build → Test → Optimize

This module is the COMPLETE Level 3 implementation. It drives the full
autonomous engineering pipeline — from specification to finished, optimized
system. The LLM (Opus 4.6+) provides the intelligence; this module provides
the structure, state machine, safety rails, and integration glue.

Components:
1. DomainResearcher — Structured knowledge acquisition before coding
2. Architect — System design, component planning, dependency ordering
3. Bootstrapper — Project creation from architecture plan
4. ReportGenerator — Generates final research reports from collected data
5. Orchestrator — Full pipeline that ties everything together

Usage:
    from engine.autonomous import Orchestrator

    orch = Orchestrator(spec="Build a REST API for task management")
    orch.run()  # Runs the full Level 3 pipeline

    # Or step-by-step:
    action = orch.get_next_action()   # What should the agent do?
    orch.run_phase()                   # Execute current phase
    orch.run_phase("build")            # Execute a specific phase
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

DR_DIR = Path(".deepresearch")


# ════════════════════════════════════════════════════════════
# 1. DOMAIN RESEARCHER — Learn before building
# ════════════════════════════════════════════════════════════

class DomainResearcher:
    """
    Structures domain research BEFORE any code is written.
    
    The LLM fills in this structure using its knowledge + web search.
    This module provides the FRAMEWORK for research, not the research itself.
    
    Output: .deepresearch/research/domain_knowledge.json
    """

    RESEARCH_PHASES = [
        {
            "phase": "understand_spec",
            "prompt": """Analyze the specification and answer:
1. What is the INPUT to this system? (data format, source, size, frequency)
2. What is the OUTPUT? (format, destination, quality requirements, latency)
3. What are the CONSTRAINTS? (language, platform, memory, time, dependencies)
4. What is the PRIMARY METRIC? (how do we measure "good enough"?)
5. What are SECONDARY METRICS? (things that should not regress)
6. What are the EDGE CASES? (unusual inputs, failure modes, boundary conditions)""",
        },
        {
            "phase": "survey_existing",
            "prompt": """Survey existing solutions in this domain:
1. What open-source implementations exist? (name, URL, approach, quality)
2. What is the STANDARD APPROACH that gets 80% of the way?
3. What are the KEY ALGORITHMS used by best implementations?
4. What are common PITFALLS that naive implementations hit?
5. What LIBRARIES exist that we should use instead of building from scratch?
6. What is the STATE OF THE ART? What makes the best implementations best?""",
        },
        {
            "phase": "identify_architecture",
            "prompt": """Based on the spec and existing solutions, define the architecture:
1. What are the CORE COMPONENTS? (list with 1-sentence purpose each)
2. What is the DATA FLOW? (input → processing → output, with intermediate steps)
3. What DESIGN PATTERNS apply? (factory, observer, strategy, etc.)
4. What is the INTERFACE between components? (function signatures, data formats)
5. What can be DEFERRED? (nice-to-have features, optimizations for later)
6. What must be built FIRST? (dependency order of components)""",
        },
        {
            "phase": "plan_testing",
            "prompt": """Design the test strategy:
1. What are the CORRECTNESS TESTS? (does it produce the right output?)
2. What are the PERFORMANCE TESTS? (how fast, how much memory?)
3. What are the STRESS TESTS? (edge cases, large inputs, concurrent access?)
4. What is the REGRESSION TEST? (what must never break as we optimize?)
5. How do we MEASURE the primary metric automatically?
6. What is the MINIMAL TEST that proves the system fundamentally works?""",
        },
    ]

    def __init__(self):
        self.research_dir = DR_DIR / "research"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge = {
            "spec_analysis": {},
            "existing_solutions": {},
            "architecture": {},
            "test_strategy": {},
            "research_complete": False,
        }

    def load(self):
        path = self.research_dir / "domain_knowledge.json"
        if path.exists():
            self.knowledge = json.loads(path.read_text())

    def save(self):
        path = self.research_dir / "domain_knowledge.json"
        path.write_text(json.dumps(self.knowledge, indent=2))

    def get_current_phase(self) -> Optional[dict]:
        """Get the next research phase that hasn't been completed."""
        phase_to_key = {
            "understand_spec": "spec_analysis",
            "survey_existing": "existing_solutions",
            "identify_architecture": "architecture",
            "plan_testing": "test_strategy",
        }
        for phase in self.RESEARCH_PHASES:
            key = phase_to_key[phase["phase"]]
            if not self.knowledge.get(key):
                return phase
        return None  # all phases complete

    def complete_phase(self, phase_name: str, findings: dict):
        """Record findings for a research phase."""
        phase_to_key = {
            "understand_spec": "spec_analysis",
            "survey_existing": "existing_solutions",
            "identify_architecture": "architecture",
            "plan_testing": "test_strategy",
        }
        key = phase_to_key.get(phase_name)
        if key:
            self.knowledge[key] = findings
            self.knowledge[key]["completed_at"] = datetime.now().isoformat()

        # Check if all phases complete
        if all(self.knowledge.get(k) for k in phase_to_key.values()):
            self.knowledge["research_complete"] = True
        self.save()

    def generate_research_report(self) -> str:
        """Generate a human-readable research report."""
        lines = ["# Domain Research Report", f"Generated: {datetime.now().isoformat()}", ""]

        if self.knowledge.get("spec_analysis"):
            lines.append("## Specification Analysis")
            for k, v in self.knowledge["spec_analysis"].items():
                if k != "completed_at":
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        if self.knowledge.get("existing_solutions"):
            lines.append("## Existing Solutions")
            for k, v in self.knowledge["existing_solutions"].items():
                if k != "completed_at":
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        if self.knowledge.get("architecture"):
            lines.append("## Architecture Decisions")
            for k, v in self.knowledge["architecture"].items():
                if k != "completed_at":
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        if self.knowledge.get("test_strategy"):
            lines.append("## Test Strategy")
            for k, v in self.knowledge["test_strategy"].items():
                if k != "completed_at":
                    lines.append(f"- **{k}**: {v}")

        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 2. ARCHITECT — Design before code
# ════════════════════════════════════════════════════════════

@dataclass
class Component:
    """A component in the system architecture."""
    name: str
    purpose: str
    files: list = field(default_factory=list)
    depends_on: list = field(default_factory=list)
    interfaces: list = field(default_factory=list)
    test_file: str = ""
    status: str = "planned"  # planned, in_progress, implemented, tested, optimized
    estimated_experiments: int = 5


class Architect:
    """
    Plans the system architecture before any code is written.
    
    Outputs:
    - Component list with dependency ordering
    - Implementation plan (what to build first)
    - Interface definitions between components
    - Test plan per component
    """

    def __init__(self):
        self.plan_path = DR_DIR / "architecture_plan.json"
        self.components = []

    def load(self):
        if self.plan_path.exists():
            data = json.loads(self.plan_path.read_text())
            self.components = [Component(**c) for c in data.get("components", [])]

    def save(self):
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "created_at": datetime.now().isoformat(),
            "components": [asdict(c) for c in self.components],
            "build_order": self.get_build_order(),
            "total_estimated_experiments": sum(c.estimated_experiments for c in self.components),
        }
        self.plan_path.write_text(json.dumps(data, indent=2))

    def add_component(self, name: str, purpose: str, depends_on: list = None,
                      files: list = None, test_file: str = "",
                      estimated_experiments: int = 5) -> Component:
        """Add a component to the architecture."""
        c = Component(
            name=name, purpose=purpose,
            depends_on=depends_on or [],
            files=files or [],
            test_file=test_file,
            estimated_experiments=estimated_experiments,
        )
        self.components.append(c)
        return c

    def get_build_order(self) -> list:
        """
        Topological sort of components by dependencies.
        Components with no dependencies come first.
        """
        # Build adjacency
        dep_map = {c.name: set(c.depends_on) for c in self.components}
        all_names = [c.name for c in self.components]
        order = []
        resolved = set()

        max_iterations = len(all_names) * 2
        iteration = 0
        while len(order) < len(all_names) and iteration < max_iterations:
            iteration += 1
            for name in all_names:
                if name in resolved:
                    continue
                deps = dep_map.get(name, set())
                if deps.issubset(resolved):
                    order.append(name)
                    resolved.add(name)

        # Add any remaining (circular deps)
        for name in all_names:
            if name not in resolved:
                order.append(name)

        return order

    def next_component(self) -> Optional[Component]:
        """Get the next component to implement (first non-implemented in build order)."""
        build_order = self.get_build_order()
        status_priority = {"planned", "in_progress"}
        for name in build_order:
            for c in self.components:
                if c.name == name and c.status in status_priority:
                    return c
        return None

    def update_status(self, component_name: str, status: str):
        """Update a component's status."""
        for c in self.components:
            if c.name == component_name:
                c.status = status
                break
        self.save()

    def progress_report(self) -> str:
        """Generate architecture progress report."""
        lines = [f"{'═'*60}", "  Architecture Progress", f"{'═'*60}"]

        status_icons = {
            "planned": "⬜", "in_progress": "🔶",
            "implemented": "🟦", "tested": "✅", "optimized": "🟢"
        }

        build_order = self.get_build_order()
        for name in build_order:
            for c in self.components:
                if c.name == name:
                    icon = status_icons.get(c.status, "❓")
                    deps = f" (needs: {', '.join(c.depends_on)})" if c.depends_on else ""
                    lines.append(f"  {icon} {c.name}: {c.purpose}{deps}")
                    lines.append(f"       Status: {c.status} | Files: {c.files} | ~{c.estimated_experiments} experiments")

        # Summary
        total = len(self.components)
        done = sum(1 for c in self.components if c.status in ("tested", "optimized"))
        lines.append(f"\n  Progress: {done}/{total} components complete")
        lines.append(f"{'═'*60}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 3. BOOTSTRAPPER — Create project structure from plan
# ════════════════════════════════════════════════════════════

class Bootstrapper:
    """
    Creates the initial project structure from an architecture plan.
    
    Does NOT write implementation code — that's the agent's job.
    Creates: directory structure, empty files with docstrings, test stubs,
    config files, and the DeepResearch state directory.
    """

    @staticmethod
    def bootstrap(architect: Architect, project_root: str = ".",
                  language: str = "python") -> dict:
        """
        Create project structure from architecture plan.
        Returns dict of created files.
        """
        root = Path(project_root)
        created = {}

        # Create .deepresearch directory
        dr = root / ".deepresearch"
        for d in ["memos", "reports", "research", "backups", "populations"]:
            (dr / d).mkdir(parents=True, exist_ok=True)

        # Create source files for each component
        for comp in architect.components:
            for f in comp.files:
                filepath = root / f
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if not filepath.exists():
                    if language == "python" and f.endswith(".py"):
                        content = f'"""\n{comp.name}: {comp.purpose}\n\nTODO: Implement this component.\n"""\n'
                        if comp.depends_on:
                            content += f"\n# Dependencies: {', '.join(comp.depends_on)}\n"
                    else:
                        content = f"// {comp.name}: {comp.purpose}\n// TODO: Implement\n"
                    filepath.write_text(content)
                    created[f] = "stub"

            # Create test file
            if comp.test_file:
                test_path = root / comp.test_file
                test_path.parent.mkdir(parents=True, exist_ok=True)
                if not test_path.exists():
                    if language == "python":
                        content = (
                            f'"""Tests for {comp.name}"""\n\n'
                            f'def test_{comp.name.lower().replace(" ", "_")}_exists():\n'
                            f'    """Smoke test: component can be imported."""\n'
                            f'    # TODO: Implement real tests\n'
                            f'    assert True\n'
                        )
                    else:
                        content = f"// Tests for {comp.name}\n// TODO: Implement\n"
                    test_path.write_text(content)
                    created[comp.test_file] = "test_stub"

        # Create config.json
        config = {
            "target_files": list(set(f for c in architect.components for f in c.files)),
            "read_only_files": list(set(c.test_file for c in architect.components if c.test_file)),
            "test_command": "pytest tests/ -q" if language == "python" else "npm test",
            "metric": "primary_metric",
            "metric_direction": "higher",
            "mutation_levels": [1, 2, 3],
            "budget_seconds": 60,
            "mutation_categories": ["structural_addition", "structural_replacement",
                                    "parametric", "integration"],
        }
        config_path = dr / "config.json"
        config_path.write_text(json.dumps(config, indent=2))
        created[".deepresearch/config.json"] = "config"

        return created


# ════════════════════════════════════════════════════════════
# 4. REPORT GENERATOR — Build final report from collected data
# ════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Generates a comprehensive research report from all collected data:
    - Domain research findings
    - Architecture decisions
    - Experiment log (kept/reverted/crashed)
    - Curriculum progress
    - Technique library results
    - Timing and efficiency stats
    """

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.dr_dir = self.root / ".deepresearch"

    def _load_json(self, path: Path, default=None):
        if path.exists():
            return json.loads(path.read_text())
        return default or {}

    def _load_jsonl(self, path: Path) -> list:
        if not path.exists():
            return []
        lines = path.read_text().strip().split("\n")
        return [json.loads(l) for l in lines if l.strip()]

    def generate(self) -> str:
        """Generate the full research report as markdown."""
        state = self._load_json(self.dr_dir / "orchestrator_state.json")
        research = self._load_json(self.dr_dir / "research" / "domain_knowledge.json")
        arch_plan = self._load_json(self.dr_dir / "architecture_plan.json")
        experiments = self._load_jsonl(self.dr_dir / "experiments.jsonl")
        curriculum = self._load_json(self.dr_dir / "curriculum.json")
        curriculum_history = self._load_jsonl(self.dr_dir / "curriculum_history.jsonl")
        strategy = self._load_json(self.dr_dir / "strategy-state.json")
        techniques = self._load_json(self.dr_dir / "research" / "techniques.json")

        lines = []

        # Header
        spec = state.get("spec", "")
        lines.append(f"# DeepResearch Level 3 — Final Report")
        lines.append(f"")
        lines.append(f"**Specification:** {spec}")
        lines.append(f"**Started:** {state.get('started_at', '?')}")
        lines.append(f"**Generated:** {datetime.now().isoformat()}")
        lines.append(f"**Total experiments:** {len(experiments)}")
        lines.append("")

        # Phase timeline
        lines.append("## Phase Timeline")
        lines.append("")
        for h in state.get("phase_history", []):
            lines.append(f"- **{h['phase'].upper()}** completed at {h['completed_at']}")
        current = state.get("current_phase", "?")
        if current != "complete":
            lines.append(f"- **{current.upper()}** (in progress)")
        lines.append("")

        # Domain Research
        if research and research.get("research_complete"):
            lines.append("## Domain Research Findings")
            lines.append("")
            for section in ["spec_analysis", "existing_solutions", "architecture", "test_strategy"]:
                data = research.get(section, {})
                if data:
                    lines.append(f"### {section.replace('_', ' ').title()}")
                    for k, v in data.items():
                        if k != "completed_at":
                            lines.append(f"- **{k}:** {v}")
                    lines.append("")

        # Architecture
        if arch_plan.get("components"):
            lines.append("## Architecture")
            lines.append("")
            lines.append(f"**Build order:** {' -> '.join(arch_plan.get('build_order', []))}")
            lines.append(f"**Estimated experiments:** {arch_plan.get('total_estimated_experiments', '?')}")
            lines.append("")
            for comp in arch_plan["components"]:
                status_icon = {"tested": "pass", "optimized": "pass", "implemented": "built",
                               "in_progress": "wip", "planned": "todo"}.get(comp.get("status", ""), "?")
                lines.append(f"- **{comp['name']}** [{status_icon}]: {comp['purpose']}")
                lines.append(f"  Files: {comp.get('files', [])}")
            lines.append("")

        # Experiment Summary
        if experiments:
            lines.append("## Experiment Summary")
            lines.append("")
            total = len(experiments)
            kept = sum(1 for e in experiments if e.get("status") == "kept")
            reverted = sum(1 for e in experiments if e.get("status") == "reverted")
            crashed = sum(1 for e in experiments if e.get("status") == "crashed")
            blocked = sum(1 for e in experiments if e.get("status") == "blocked")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Total experiments | {total} |")
            lines.append(f"| Kept | {kept} ({kept/total*100:.0f}%) |")
            lines.append(f"| Reverted | {reverted} ({reverted/total*100:.0f}%) |")
            lines.append(f"| Crashed | {crashed} |")
            lines.append(f"| Blocked | {blocked} |")
            lines.append(f"| Success rate | {kept/total*100:.1f}% |")
            lines.append("")

            # Mutation type breakdown
            mt_stats = {}
            for e in experiments:
                mt = e.get("mutation_type", "unknown")
                if mt not in mt_stats:
                    mt_stats[mt] = {"total": 0, "kept": 0}
                mt_stats[mt]["total"] += 1
                if e.get("status") == "kept":
                    mt_stats[mt]["kept"] += 1
            lines.append("### By Mutation Type")
            lines.append("")
            lines.append("| Type | Total | Kept | Rate |")
            lines.append("|------|-------|------|------|")
            for mt, stats in sorted(mt_stats.items(), key=lambda x: -x[1]["total"]):
                rate = stats["kept"] / stats["total"] * 100 if stats["total"] > 0 else 0
                lines.append(f"| {mt} | {stats['total']} | {stats['kept']} | {rate:.0f}% |")
            lines.append("")

            # Top improvements
            improvements = sorted(
                [e for e in experiments if e.get("improvement_pct", 0) > 0],
                key=lambda e: -e.get("improvement_pct", 0)
            )[:10]
            if improvements:
                lines.append("### Top Improvements")
                lines.append("")
                for e in improvements:
                    lines.append(
                        f"- **{e.get('id')}** ({e.get('mutation_type')}): "
                        f"+{e.get('improvement_pct', 0):.2f}% — {e.get('description', '')}"
                    )
                lines.append("")

        # Curriculum Progress
        if curriculum.get("stages"):
            lines.append("## Curriculum Progress")
            lines.append("")
            for i, stage in enumerate(curriculum["stages"]):
                # Check if completed from history
                completed = any(
                    h.get("event") == "stage_completed" and h.get("stage") == stage["name"]
                    for h in curriculum_history
                )
                icon = "pass" if completed else "pending"
                lines.append(f"- [{icon}] **{stage['name']}**: {stage.get('description', '')}")
                lines.append(f"  Target: {stage['metric']} {'>=' if stage.get('direction') == 'higher' else '<='} {stage['target']}")
            lines.append("")

        # Techniques
        tech_list = techniques.get("techniques", [])
        if tech_list:
            lines.append("## Technique Library")
            lines.append("")
            successful = [t for t in tech_list if t.get("tried") and "worked" in t.get("result", "").lower()]
            failed = [t for t in tech_list if t.get("tried") and "failed" in t.get("result", "").lower()]
            untried = [t for t in tech_list if not t.get("tried")]
            if successful:
                lines.append(f"### Successful ({len(successful)})")
                for t in successful:
                    lines.append(f"- **{t['name']}**: {t.get('result', '')}")
            if failed:
                lines.append(f"### Failed ({len(failed)})")
                for t in failed:
                    lines.append(f"- **{t['name']}**: {t.get('result', '')}")
            if untried:
                lines.append(f"### Untried ({len(untried)})")
                for t in untried:
                    lines.append(f"- **{t['name']}**: {t.get('description', '')} [{t.get('complexity', '?')}]")
            lines.append("")

        # Strategy stats
        if strategy:
            lines.append("## Strategy Engine Stats")
            lines.append("")
            lines.append(f"- **Final temperature:** {strategy.get('temperature', '?')}")
            lines.append(f"- **Best metric:** {strategy.get('best_metric', 'N/A')}")
            lines.append(f"- **Baseline metric:** {strategy.get('baseline_metric', 'N/A')}")
            baseline = strategy.get("baseline_metric")
            best = strategy.get("best_metric")
            if baseline is not None and best is not None and baseline != 0:
                improvement = abs(best - baseline) / abs(baseline) * 100
                lines.append(f"- **Total improvement:** {improvement:.1f}%")
            lines.append("")

            # Bandit arms
            arms = strategy.get("bandit_arms", {})
            if arms:
                lines.append("### Bandit Arms (Thompson Sampling)")
                lines.append("")
                lines.append("| Mutation Type | Trials | Successes | Rate |")
                lines.append("|--------------|--------|-----------|------|")
                for name, arm in sorted(arms.items(), key=lambda x: -x[1].get("trials", 0)):
                    trials = arm.get("trials", 0)
                    successes = arm.get("alpha", 1) - 1
                    rate = successes / trials * 100 if trials > 0 else 0
                    lines.append(f"| {name} | {trials} | {successes} | {rate:.0f}% |")
                lines.append("")

        return "\n".join(lines)

    def save(self, filename: str = "research_report.md") -> str:
        """Generate and save the report. Returns the file path."""
        report = self.generate()
        report_dir = self.dr_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = report_dir / f"{ts}_{filename}"
        path.write_text(report)
        return str(path)


# ════════════════════════════════════════════════════════════
# 5. ORCHESTRATOR — The full Level 3 pipeline
# ════════════════════════════════════════════════════════════

class Orchestrator:
    """
    The full Level 3 pipeline:
    
    Phase 0: RESEARCH — Understand the domain before writing code
    Phase 1: ARCHITECT — Design the system, plan components
    Phase 2: BOOTSTRAP — Create project structure
    Phase 3: BUILD — Implement components (using Level 2 mutations)
    Phase 4: TEST — Verify correctness (curriculum stage 1)
    Phase 5: OPTIMIZE — Improve performance (curriculum stages 2+)
    Phase 6: REPORT — Document findings
    
    The Orchestrator doesn't DO the work — it tells the LLM agent
    WHAT to do next. The agent uses its intelligence to actually
    write code, make decisions, and form hypotheses.
    """

    PHASES = [
        {"name": "research", "description": "Understand the domain and existing solutions"},
        {"name": "architect", "description": "Design system components and their interfaces"},
        {"name": "bootstrap", "description": "Create project structure and test stubs"},
        {"name": "build", "description": "Implement components in dependency order"},
        {"name": "test", "description": "Verify correctness via curriculum stage 1"},
        {"name": "optimize", "description": "Improve metrics via curriculum stages 2+"},
        {"name": "report", "description": "Document findings and generate research report"},
    ]

    PHASE_PREREQUISITES = {
        "research": [],
        "architect": ["research"],
        "bootstrap": ["architect"],
        "build": ["bootstrap"],
        "test": ["build"],
        "optimize": ["test"],
        "report": [],  # can generate report at any time
    }

    def __init__(self, spec: str = "", project_root: str = "."):
        self.spec = spec
        self.root = Path(project_root)
        self.state_path = DR_DIR / "orchestrator_state.json"
        self.researcher = DomainResearcher()
        self.architect = Architect()
        self.report_gen = ReportGenerator(project_root)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {
            "spec": self.spec,
            "current_phase": "research",
            "phase_history": [],
            "started_at": datetime.now().isoformat(),
            "total_experiments": 0,
        }

    def save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    @property
    def current_phase(self) -> str:
        return self.state.get("current_phase", "research")

    def advance_phase(self):
        """Move to the next phase."""
        phase_names = [p["name"] for p in self.PHASES]
        idx = phase_names.index(self.current_phase)
        self.state["phase_history"].append({
            "phase": self.current_phase,
            "completed_at": datetime.now().isoformat(),
        })
        if idx + 1 < len(phase_names):
            self.state["current_phase"] = phase_names[idx + 1]
        else:
            self.state["current_phase"] = "complete"
        self.save_state()

    def get_next_action(self) -> dict:
        """
        Get the next action the agent should take.
        
        Returns a structured instruction that tells the agent:
        - What to do
        - What inputs are available
        - What output is expected
        - What tools to use
        """
        phase = self.current_phase

        if phase == "research":
            self.researcher.load()
            research_phase = self.researcher.get_current_phase()
            if research_phase is None:
                return {"action": "advance_phase",
                        "message": "Research complete. Run orchestrator.advance_phase()"}
            return {
                "action": "research",
                "phase": research_phase["phase"],
                "prompt": research_phase["prompt"],
                "spec": self.spec,
                "tools": ["web_search", "read_files"],
                "output": "Call researcher.complete_phase() with structured findings",
            }

        elif phase == "architect":
            self.architect.load()
            if not self.architect.components:
                return {
                    "action": "design_architecture",
                    "prompt": (
                        "Based on the domain research, design the system architecture.\n"
                        "For each component, call architect.add_component() with:\n"
                        "- name, purpose, files (list of file paths), depends_on (list of component names),\n"
                        "  estimated_experiments (how many experiments to implement it)\n"
                        "Then call architect.save() and orchestrator.advance_phase()"
                    ),
                    "research": self.researcher.knowledge,
                    "tools": ["architect.add_component", "architect.save"],
                }
            return {"action": "advance_phase",
                    "message": "Architecture defined. Run orchestrator.advance_phase()"}

        elif phase == "bootstrap":
            return {
                "action": "bootstrap_project",
                "prompt": "Create the project structure from the architecture plan.",
                "tools": ["Bootstrapper.bootstrap(architect)"],
                "output": "Call Bootstrapper.bootstrap() then orchestrator.advance_phase()",
            }

        elif phase == "build":
            self.architect.load()
            next_comp = self.architect.next_component()
            if next_comp is None:
                return {"action": "advance_phase",
                        "message": "All components implemented. Run orchestrator.advance_phase()"}
            return {
                "action": "implement_component",
                "component": asdict(next_comp),
                "prompt": (
                    f"Implement component '{next_comp.name}': {next_comp.purpose}\n"
                    f"Files to create/modify: {next_comp.files}\n"
                    f"Dependencies (already implemented): {next_comp.depends_on}\n"
                    f"Test file: {next_comp.test_file}\n\n"
                    "Use the DeepResearch experiment loop:\n"
                    "1. Deep Read the dependencies to understand the interfaces\n"
                    "2. Form hypothesis about the best implementation approach\n"
                    "3. Write the code (structural_addition mutation)\n"
                    "4. Run tests to verify correctness\n"
                    "5. Reflect on what worked and what didn't\n"
                    "6. When tests pass, update component status to 'tested'"
                ),
                "mutation_type": "structural_addition",
                "tools": ["mutations.propose", "mutations.execute", "architect.update_status"],
            }

        elif phase == "test":
            return {
                "action": "verify_correctness",
                "prompt": (
                    "Run the full test suite. If any tests fail, fix them using the\n"
                    "DeepResearch experiment loop. This is curriculum Stage 1.\n"
                    "Advance when all tests pass."
                ),
                "curriculum_stage": 1,
                "tools": ["curriculum.check_advancement"],
            }

        elif phase == "optimize":
            return {
                "action": "optimize",
                "prompt": (
                    "Use the DeepResearch experiment loop to optimize the system.\n"
                    "Check the curriculum for the current target metric.\n"
                    "Use Level 1 (parametric) and Level 2 (structural) mutations.\n"
                    "Advance when the curriculum stage target is met.\n"
                    "Continue through all curriculum stages."
                ),
                "tools": ["curriculum.current_stage", "mutations", "strategy"],
            }

        elif phase == "report":
            return {
                "action": "generate_report",
                "prompt": (
                    "Generate a comprehensive research report:\n"
                    "1. What was built (architecture overview)\n"
                    "2. Key decisions and why (from research memos)\n"
                    "3. What worked and what didn't (from experiment log)\n"
                    "4. Performance metrics (from curriculum)\n"
                    "5. Recommendations for future improvement"
                ),
                "tools": ["strategy.py report", "curriculum.progress_report"],
            }

        else:
            return {"action": "complete",
                    "message": "All phases complete. The system is built and optimized."}

    def status_report(self) -> str:
        """Full status of the Level 3 pipeline."""
        lines = [
            f"{'═'*60}",
            f"  Level 3 Orchestrator — Pipeline Status",
            f"{'═'*60}",
            f"  Spec: {self.spec[:80]}..." if len(self.spec) > 80 else f"  Spec: {self.spec}",
            f"  Started: {self.state.get('started_at', '?')}",
            f"  Current phase: {self.current_phase}",
            f"  Experiments: {self.state.get('total_experiments', 0)}",
            "",
        ]

        phase_icons = {"complete": "✅", "research": "🔬", "architect": "📐",
                       "bootstrap": "🏗️", "build": "🔨", "test": "🧪",
                       "optimize": "⚡", "report": "📝"}

        for p in self.PHASES:
            completed = any(h["phase"] == p["name"] for h in self.state.get("phase_history", []))
            is_current = p["name"] == self.current_phase
            if completed:
                icon = "✅"
            elif is_current:
                icon = "🔶"
            else:
                icon = "⬜"
            name = p["name"].upper()
            lines.append(f"  {icon} {name}: {p['description']}")

        lines.append(f"\n{'═'*60}")
        return "\n".join(lines)

    # ── Phase Validation ──────────────────────────────────────

    def validate_phase(self, phase_name: str) -> dict:
        """
        Check if a phase can be executed.
        Returns {"valid": bool, "reasons": list, "warnings": list}.
        """
        result = {"valid": True, "reasons": [], "warnings": []}

        # Check phase exists
        phase_names = [p["name"] for p in self.PHASES]
        if phase_name not in phase_names:
            result["valid"] = False
            result["reasons"].append(f"Unknown phase: {phase_name}")
            return result

        # Check prerequisites
        completed_phases = {h["phase"] for h in self.state.get("phase_history", [])}
        for prereq in self.PHASE_PREREQUISITES.get(phase_name, []):
            if prereq not in completed_phases:
                result["valid"] = False
                result["reasons"].append(
                    f"Phase '{phase_name}' requires '{prereq}' to be completed first"
                )

        # Phase-specific validation
        if phase_name == "architect":
            self.researcher.load()
            if not self.researcher.knowledge.get("research_complete"):
                result["warnings"].append(
                    "Research is not marked as complete — architecture may be uninformed"
                )

        elif phase_name == "bootstrap":
            self.architect.load()
            if not self.architect.components:
                result["valid"] = False
                result["reasons"].append("No architecture plan defined")

        elif phase_name == "build":
            self.architect.load()
            if not self.architect.components:
                result["valid"] = False
                result["reasons"].append("No components to build")

        return result

    # ── Run Methods ───────────────────────────────────────────

    def run(self, max_experiments: int = 200) -> dict:
        """
        Run the full Level 3 pipeline from current phase to completion.

        This is the main entry point. It drives through all phases:
        research → architect → bootstrap → build → test → optimize → report

        Args:
            max_experiments: Safety limit on total experiments (build + test + optimize).

        Returns:
            {"status": "complete"/"stopped"/"error", "phases_completed": list,
             "total_experiments": int, "report_path": str}
        """
        result = {
            "status": "complete",
            "phases_completed": [],
            "total_experiments": 0,
            "report_path": "",
            "errors": [],
        }

        while self.current_phase != "complete":
            # Safety limit
            total_exp = self.state.get("total_experiments", 0)
            if total_exp >= max_experiments:
                result["status"] = "stopped"
                result["errors"].append(
                    f"Experiment budget exhausted ({total_exp}/{max_experiments})"
                )
                break

            phase = self.current_phase
            validation = self.validate_phase(phase)
            if not validation["valid"]:
                result["status"] = "error"
                result["errors"].extend(validation["reasons"])
                break

            phase_result = self.run_phase(phase)
            if phase_result.get("status") == "error":
                result["status"] = "error"
                result["errors"].append(
                    f"Phase '{phase}' failed: {phase_result.get('error', '?')}"
                )
                break

            result["phases_completed"].append(phase)
            result["total_experiments"] = self.state.get("total_experiments", 0)

        result["total_experiments"] = self.state.get("total_experiments", 0)
        return result

    def run_phase(self, phase_name: str = None) -> dict:
        """
        Execute a single phase of the pipeline.

        If phase_name is None, runs the current phase.
        Returns {"status": "complete"/"error", "phase": str, ...}
        """
        phase = phase_name or self.current_phase

        if phase == "complete":
            return {"status": "complete", "phase": "complete",
                    "message": "Pipeline already complete"}

        # Validate
        validation = self.validate_phase(phase)
        if not validation["valid"]:
            return {"status": "error", "phase": phase,
                    "error": "; ".join(validation["reasons"])}

        # Dispatch to phase handler
        handlers = {
            "research": self._run_research,
            "architect": self._run_architect,
            "bootstrap": self._run_bootstrap,
            "build": self._run_build,
            "test": self._run_test,
            "optimize": self._run_optimize,
            "report": self._run_report,
        }
        handler = handlers.get(phase)
        if not handler:
            return {"status": "error", "phase": phase,
                    "error": f"No handler for phase '{phase}'"}

        result = handler()
        result["phase"] = phase

        # Auto-advance if handler completed successfully
        if result.get("status") == "complete" and self.current_phase == phase:
            self.advance_phase()

        return result

    def _run_research(self) -> dict:
        """Execute research phase — iterate through all 4 research sub-phases."""
        self.researcher.load()
        phases_done = []

        while True:
            phase = self.researcher.get_current_phase()
            if phase is None:
                break
            phases_done.append(phase["phase"])
            # Return structured instruction for the agent
            # The agent fills in the findings using researcher.complete_phase()
            return {
                "status": "needs_agent",
                "action": "research",
                "research_phase": phase["phase"],
                "prompt": phase["prompt"],
                "spec": self.spec or self.state.get("spec", ""),
                "instruction": (
                    f"Complete research phase '{phase['phase']}'.\n\n"
                    f"{phase['prompt']}\n\n"
                    f"After answering, call:\n"
                    f"  orchestrator.researcher.complete_phase('{phase['phase']}', findings)\n"
                    f"where findings is a dict with your answers.\n"
                    f"Then call orchestrator.run_phase('research') again for the next phase."
                ),
                "phases_done": phases_done,
            }

        return {
            "status": "complete",
            "phases_done": phases_done,
            "report": self.researcher.generate_research_report(),
        }

    def _run_architect(self) -> dict:
        """Execute architect phase."""
        self.architect.load()

        if self.architect.components:
            return {
                "status": "complete",
                "components": len(self.architect.components),
                "build_order": self.architect.get_build_order(),
            }

        self.researcher.load()
        return {
            "status": "needs_agent",
            "action": "design_architecture",
            "research": self.researcher.knowledge,
            "instruction": (
                "Design the system architecture based on domain research.\n\n"
                "For each component, call:\n"
                "  orchestrator.architect.add_component(\n"
                "      name='component_name',\n"
                "      purpose='what it does',\n"
                "      files=['src/file.py'],\n"
                "      depends_on=['other_component'],\n"
                "      test_file='tests/test_component.py',\n"
                "      estimated_experiments=5\n"
                "  )\n\n"
                "Then call:\n"
                "  orchestrator.architect.save()\n"
                "  orchestrator.run_phase('architect')  # to verify and advance"
            ),
        }

    def _run_bootstrap(self) -> dict:
        """Execute bootstrap phase — create project structure."""
        self.architect.load()
        created = Bootstrapper.bootstrap(self.architect, str(self.root))
        return {
            "status": "complete",
            "files_created": created,
            "file_count": len(created),
        }

    def _run_build(self) -> dict:
        """
        Execute build phase — implement components in dependency order.

        Iterates through components. For each unbuilt component, returns
        instructions for the agent. The agent implements it, then calls
        run_phase('build') again for the next component.
        """
        self.architect.load()
        next_comp = self.architect.next_component()

        if next_comp is None:
            return {
                "status": "complete",
                "message": "All components implemented",
                "components_built": [
                    c.name for c in self.architect.components
                    if c.status in ("implemented", "tested", "optimized")
                ],
            }

        # Mark as in_progress
        self.architect.update_status(next_comp.name, "in_progress")

        # Build order progress
        build_order = self.architect.get_build_order()
        done = [c for c in self.architect.components
                if c.status in ("implemented", "tested", "optimized")]
        progress = f"{len(done)}/{len(self.architect.components)}"

        return {
            "status": "needs_agent",
            "action": "implement_component",
            "component": asdict(next_comp),
            "build_progress": progress,
            "build_order": build_order,
            "instruction": (
                f"Implement component '{next_comp.name}': {next_comp.purpose}\n\n"
                f"Progress: {progress} components built\n"
                f"Files to create/modify: {next_comp.files}\n"
                f"Dependencies (already implemented): {next_comp.depends_on}\n"
                f"Test file: {next_comp.test_file}\n\n"
                "Use the DeepResearch experiment loop:\n"
                "1. R1 DEEP READ: Read the dependency files to understand interfaces\n"
                "2. R2 HYPOTHESIZE: What is the best implementation approach?\n"
                "3. R3 PREDICT: How many experiments will this take?\n"
                "4. IMPLEMENT: Write the code (structural_addition mutation)\n"
                "5. TEST: Run tests to verify correctness\n"
                "6. REFLECT: What worked, what didn't?\n\n"
                "When tests pass, call:\n"
                f"  orchestrator.architect.update_status('{next_comp.name}', 'tested')\n"
                f"  orchestrator.record_experiment('build', '{next_comp.name}')\n"
                "  orchestrator.run_phase('build')  # next component"
            ),
            "mutation_type": "structural_addition",
        }

    def _run_test(self) -> dict:
        """
        Execute test phase — verify all components pass tests.

        This is curriculum Stage 1 (correctness). The agent runs the
        full test suite and fixes any failures using the experiment loop.
        """
        from engine.curriculum import CurriculumRunner
        curriculum = CurriculumRunner()
        stage = curriculum.current_stage()

        # Check if we have a test command
        config_path = DR_DIR / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())

        if not config.get("test_command"):
            return {
                "status": "needs_agent",
                "action": "configure_tests",
                "instruction": (
                    "No test_command configured. Set it in .deepresearch/config.json:\n"
                    '  {"test_command": "pytest tests/ -q"}\n\n'
                    "Then call orchestrator.run_phase('test') again."
                ),
            }

        return {
            "status": "needs_agent",
            "action": "verify_correctness",
            "curriculum_stage": stage,
            "test_command": config.get("test_command"),
            "instruction": (
                "Run the full test suite and fix any failures.\n\n"
                f"Test command: {config.get('test_command')}\n"
                f"{'Curriculum stage: ' + stage['name'] if stage else 'No curriculum defined'}\n\n"
                "For each failing test:\n"
                "1. R1 DEEP READ: Understand the test and the code it tests\n"
                "2. R2 HYPOTHESIZE: What is causing the failure?\n"
                "3. R3 PREDICT: What fix will make it pass?\n"
                "4. IMPLEMENT: Apply the fix (structural_replacement mutation)\n"
                "5. TEST: Run the test suite again\n"
                "6. REFLECT: Was the fix correct? Any side effects?\n\n"
                "When all tests pass, call:\n"
                "  orchestrator.record_experiment('test', 'all_tests_passing')\n"
                "  orchestrator.run_phase('test')  # to verify and advance\n\n"
                "If using curriculum, also call:\n"
                "  curriculum.update_metrics({'test_pass_rate': 1.0})\n"
                "  curriculum.check_advancement()"
            ),
        }

    def _run_optimize(self) -> dict:
        """
        Execute optimize phase — improve metrics through curriculum stages.

        Runs the experiment loop until all curriculum stages are met
        or the experiment budget is exhausted.
        """
        from engine.curriculum import CurriculumRunner
        curriculum = CurriculumRunner()

        if curriculum.is_complete():
            return {
                "status": "complete",
                "message": "All curriculum stages complete",
            }

        stage = curriculum.current_stage()
        strategy = curriculum.get_mutation_strategy()

        # Calculate remaining budget
        total_exp = self.state.get("total_experiments", 0)
        config_path = DR_DIR / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())
        budget = config.get("experiment_budget", 200)
        remaining = max(0, budget - total_exp)

        return {
            "status": "needs_agent",
            "action": "optimize",
            "curriculum_stage": stage,
            "mutation_strategy": strategy,
            "experiments_remaining": remaining,
            "instruction": (
                f"Optimize: {stage['name']} — {stage.get('description', '')}\n\n"
                f"Target: {stage['metric']} "
                f"{'≥' if stage.get('direction') == 'higher' else '≤'} "
                f"{stage['target']}\n"
                f"Preferred mutations: {strategy.get('preferred_types', [])}\n"
                f"Temperature: {strategy.get('temperature', 0.5)}\n"
                f"Focus areas: {strategy.get('focus_areas', [])}\n"
                f"Experiments remaining: {remaining}\n\n"
                "Run the DeepResearch experiment loop:\n"
                "1. R1 DEEP READ: What is the current bottleneck?\n"
                "2. R2 HYPOTHESIZE: What change would improve the metric?\n"
                "3. R3 PREDICT: How much improvement do you expect?\n"
                "4. IMPLEMENT: Apply ONE focused mutation\n"
                "5. EVALUATE: Measure the metric\n"
                "6. REFLECT: Was prediction correct? Update mental model.\n\n"
                "After each experiment, call:\n"
                "  orchestrator.record_experiment('optimize', description)\n"
                "  curriculum.update_metrics({...})\n"
                "  curriculum.check_advancement()\n\n"
                "When the curriculum stage target is met, call:\n"
                "  orchestrator.run_phase('optimize')  # check if more stages remain"
            ),
        }

    def _run_report(self) -> dict:
        """Execute report phase — generate the final research report."""
        report_path = self.report_gen.save()
        return {
            "status": "complete",
            "report_path": report_path,
            "message": f"Report saved to {report_path}",
        }

    # ── Experiment Tracking ──────────────────────────────────

    def record_experiment(self, phase: str, description: str,
                          status: str = "kept"):
        """Record that an experiment was performed during a phase."""
        self.state["total_experiments"] = self.state.get("total_experiments", 0) + 1
        experiments = self.state.setdefault("experiments_by_phase", {})
        phase_exp = experiments.setdefault(phase, [])
        phase_exp.append({
            "description": description,
            "status": status,
            "experiment_number": self.state["total_experiments"],
            "timestamp": datetime.now().isoformat(),
        })
        self.save_state()

    def experiments_in_phase(self, phase: str) -> int:
        """Count experiments performed in a specific phase."""
        experiments = self.state.get("experiments_by_phase", {})
        return len(experiments.get(phase, []))

    # ── Convenience Methods ──────────────────────────────────

    def reset_phase(self, phase_name: str):
        """Reset a phase to re-run it (removes from history)."""
        self.state["phase_history"] = [
            h for h in self.state.get("phase_history", [])
            if h["phase"] != phase_name
        ]
        # If we're past this phase, rewind
        phase_names = [p["name"] for p in self.PHASES]
        if phase_name in phase_names:
            current_idx = phase_names.index(self.current_phase) if self.current_phase in phase_names else len(phase_names)
            target_idx = phase_names.index(phase_name)
            if target_idx < current_idx:
                self.state["current_phase"] = phase_name
        self.save_state()

    def skip_phase(self, phase_name: str):
        """Skip a phase (mark as completed without running)."""
        self.state["phase_history"].append({
            "phase": phase_name,
            "completed_at": datetime.now().isoformat(),
            "skipped": True,
        })
        if self.current_phase == phase_name:
            self.advance_phase()
        else:
            self.save_state()

    def generate_report(self) -> str:
        """Generate the research report without changing phase state."""
        return self.report_gen.generate()

    def save_report(self, filename: str = "research_report.md") -> str:
        """Generate and save report to .deepresearch/reports/."""
        return self.report_gen.save(filename)
