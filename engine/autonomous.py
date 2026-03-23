"""
DeepResearch Level 3 — Autonomous Engineer

The full pipeline: Specification → Research → Architecture → Build → Test → Optimize

This module provides the SCAFFOLDING for autonomous software engineering.
The actual intelligence comes from the LLM (Opus 4.6+) — this module
structures HOW the LLM applies its intelligence.

Components:
1. DomainResearcher — Structured knowledge acquisition before coding
2. Architect — System design, component planning, dependency ordering
3. Bootstrapper — Project creation from architecture plan
4. Orchestrator — Full pipeline that ties everything together

Usage:
    from engine.autonomous import Orchestrator
    
    orch = Orchestrator(spec="Build a REST API for task management")
    orch.run()  # Runs the full Level 3 pipeline
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
# 4. ORCHESTRATOR — The full Level 3 pipeline
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

    def __init__(self, spec: str = "", project_root: str = "."):
        self.spec = spec
        self.root = Path(project_root)
        self.state_path = DR_DIR / "orchestrator_state.json"
        self.researcher = DomainResearcher()
        self.architect = Architect()
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
