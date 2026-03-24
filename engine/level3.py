#!/usr/bin/env python3
"""
DeepResearch Level 2-3 CLI — Unified interface for all Level 2-3 features.

Commands:
    python -m engine.level3 status              # Full pipeline status
    python -m engine.level3 research             # Show next research phase
    python -m engine.level3 architect            # Show architecture plan
    python -m engine.level3 curriculum [domain]  # Show/create curriculum
    python -m engine.level3 mutations            # List mutation types
    python -m engine.level3 next                 # What should the agent do next?
    python -m engine.level3 bootstrap            # Create project structure
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.mutations import MutationManager, MUTATION_TYPES, FeatureDiscovery
from engine.curriculum import CurriculumRunner
from engine.autonomous import Orchestrator, DomainResearcher, Architect, Bootstrapper, ReportGenerator

DR_DIR = Path(".deepresearch")


def cmd_status():
    """Show full pipeline status."""
    orch = Orchestrator()
    print(orch.status_report())

    # Show curriculum if exists
    runner = CurriculumRunner()
    if runner.stages:
        print()
        print(runner.progress_report())

    # Show architecture if exists
    arch = Architect()
    arch.load()
    if arch.components:
        print()
        print(arch.progress_report())


def cmd_research():
    """Show current research phase."""
    researcher = DomainResearcher()
    researcher.load()
    phase = researcher.get_current_phase()
    if phase is None:
        print("All research phases complete.")
        print(researcher.generate_research_report())
    else:
        print(f"Current research phase: {phase['phase']}")
        print(f"\nPrompt for the agent:\n{phase['prompt']}")


def cmd_architect():
    """Show architecture plan."""
    arch = Architect()
    arch.load()
    if not arch.components:
        print("No architecture plan defined yet.")
        print("Run the research phase first, then design the architecture.")
    else:
        print(arch.progress_report())
        print(f"\nBuild order: {' → '.join(arch.get_build_order())}")


def cmd_curriculum(domain=None):
    """Show or create curriculum."""
    if domain:
        runner = CurriculumRunner.create_from_template(domain)
        print(f"Created curriculum from '{domain}' template.")
        print(runner.progress_report())
    else:
        runner = CurriculumRunner()
        if runner.stages:
            print(runner.progress_report())
        else:
            print("No curriculum defined.")
            print(f"\nAvailable templates: {', '.join(CurriculumRunner.templates().keys())}")
            print("Create: python -m engine.level3 curriculum <domain>")


def cmd_mutations():
    """List all mutation types."""
    print(f"{'═'*60}")
    print("  Available Mutation Types")
    print(f"{'═'*60}")
    for name, mt in MUTATION_TYPES.items():
        print(f"\n  Level {mt['level']} | {name}")
        print(f"  {mt['description']}")
        print(f"  Risk: {mt['risk']} | Tests required: {mt['requires_tests']}")
        print(f"  Max files: {mt['max_files']} | Max lines: {mt['max_lines_changed']}")


def cmd_next():
    """Show what the agent should do next."""
    orch = Orchestrator()
    action = orch.get_next_action()
    print(f"Phase: {orch.current_phase}")
    print(f"Action: {action['action']}")
    if 'prompt' in action:
        print(f"\nInstructions for the agent:\n{action['prompt']}")
    if 'message' in action:
        print(f"\n{action['message']}")
    if 'tools' in action:
        print(f"\nTools to use: {action['tools']}")


def cmd_bootstrap():
    """Create project structure from architecture plan."""
    arch = Architect()
    arch.load()
    if not arch.components:
        print("No architecture plan. Run 'research' and 'architect' first.")
        return
    created = Bootstrapper.bootstrap(arch)
    print(f"Created {len(created)} files:")
    for f, ftype in created.items():
        print(f"  {ftype:12s} {f}")


def cmd_discover():
    """Show feature discovery analysis prompt."""
    print(FeatureDiscovery.generate_analysis_prompt())


def cmd_run(max_experiments=200):
    """Run the full Level 3 pipeline."""
    orch = Orchestrator()
    if not orch.spec and not orch.state.get("spec"):
        print("No spec set. Initialize first:")
        print("  python -m engine.level3 init --spec 'your specification'")
        return

    print(f"Starting Level 3 pipeline from phase: {orch.current_phase}")
    print(f"Spec: {orch.spec or orch.state.get('spec', '?')}")
    print(f"Max experiments: {max_experiments}")
    print()

    result = orch.run(max_experiments=max_experiments)

    print(f"\nPipeline result: {result['status']}")
    print(f"Phases completed: {result['phases_completed']}")
    print(f"Total experiments: {result['total_experiments']}")
    if result.get("report_path"):
        print(f"Report: {result['report_path']}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")


def cmd_run_phase(phase_name=None):
    """Run a single phase of the Level 3 pipeline."""
    orch = Orchestrator()
    phase = phase_name or orch.current_phase

    # Validate first
    validation = orch.validate_phase(phase)
    if not validation["valid"]:
        print(f"Cannot run phase '{phase}':")
        for reason in validation["reasons"]:
            print(f"  - {reason}")
        return
    if validation["warnings"]:
        for w in validation["warnings"]:
            print(f"  Warning: {w}")

    print(f"Running phase: {phase}")
    result = orch.run_phase(phase)
    print(f"Status: {result['status']}")

    # Show result details
    if result.get("instruction"):
        print(f"\nAgent instructions:\n{result['instruction']}")
    if result.get("message"):
        print(f"\n{result['message']}")
    if result.get("report_path"):
        print(f"Report saved: {result['report_path']}")
    if result.get("files_created"):
        print(f"Files created: {result['file_count']}")
        for f, ftype in result["files_created"].items():
            print(f"  {ftype:12s} {f}")
    if result.get("error"):
        print(f"Error: {result['error']}")


def cmd_report():
    """Generate research report from collected data."""
    orch = Orchestrator()
    path = orch.save_report()
    print(f"Report saved: {path}")
    print()
    # Also print to stdout
    print(orch.generate_report())


def cmd_validate(phase_name=None):
    """Validate if a phase can be run."""
    orch = Orchestrator()
    phase = phase_name or orch.current_phase
    validation = orch.validate_phase(phase)
    if validation["valid"]:
        print(f"Phase '{phase}': READY")
    else:
        print(f"Phase '{phase}': BLOCKED")
        for r in validation["reasons"]:
            print(f"  - {r}")
    for w in validation.get("warnings", []):
        print(f"  Warning: {w}")


def cmd_reset(phase_name):
    """Reset a phase to re-run it."""
    orch = Orchestrator()
    orch.reset_phase(phase_name)
    print(f"Phase '{phase_name}' reset. Current phase: {orch.current_phase}")


def cmd_skip(phase_name):
    """Skip a phase."""
    orch = Orchestrator()
    orch.skip_phase(phase_name)
    print(f"Phase '{phase_name}' skipped. Current phase: {orch.current_phase}")


def cmd_init(spec="", domain="custom"):
    """Initialize a Level 3 project."""
    DR_DIR.mkdir(parents=True, exist_ok=True)

    # Create orchestrator state
    orch = Orchestrator(spec=spec)
    orch.save_state()

    # Create curriculum
    runner = CurriculumRunner.create_from_template(domain)

    print(f"Initialized Level 3 project")
    print(f"  Spec: {spec[:80] if spec else '(not set)'}")
    print(f"  Curriculum: {domain} ({len(runner.stages)} stages)")
    print(f"\nNext step: python -m engine.level3 next")


def cmd_knowledge(domain=None, spec=None, bottleneck=None):
    """Show knowledge acquisition status and search queries."""
    from engine.knowledge import KnowledgeAcquisition
    ka = KnowledgeAcquisition(domain=domain or "general", spec=spec or "")

    # Show research summary if sources exist
    if ka.sources.sources:
        print(ka.summary())
    else:
        print("No external knowledge acquired yet.\n")

    # Show search queries
    queries = ka.generate_searches(bottleneck=bottleneck)
    print(f"Recommended search queries (domain: {ka.domain}):")
    for i, q in enumerate(queries[:10], 1):
        print(f"  {i}. [{q['priority']:.2f}] {q['query']}")

    # Show reading protocol
    if not ka.sources.sources:
        print(f"\nTo start: search these queries, read the results, and use:")
        print(f"  ka.register_source(url, title, source_type)")
        print(f"  ka.mark_source_read(url, summary, key_insights)")
        print(f"  ka.extract_technique(source_url, name, description, ...)")


def cmd_techniques():
    """Show discovered techniques."""
    from engine.knowledge import TechniqueLibrary
    lib = TechniqueLibrary()
    if not lib.techniques:
        print("No techniques discovered yet. Run domain research first.")
        return
    untried = lib.untried()
    if untried:
        print("Untried techniques (by priority):")
        for t in untried:
            print(f"  [{t.priority:.2f}] {t.name}: {t.description}")
            print(f"         Impact: {t.expected_impact} | Complexity: {t.complexity}")
    successful = lib.successful()
    if successful:
        print(f"\nSuccessful ({len(successful)}):")
        for t in successful:
            print(f"  ✅ {t.name}: {t.result}")
    failed = lib.failed()
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for t in failed:
            print(f"  ❌ {t.name}: {t.result}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("DeepResearch Level 2-3 Engine")
        print()
        print("Commands:")
        print("  init [--spec 'text'] [--domain name]   Initialize Level 3 project")
        print("  run [--max-experiments N]               Run the full L3 pipeline")
        print("  run-phase [phase]                       Run a single phase")
        print("  status                                  Full pipeline status")
        print("  next                                    What should the agent do next?")
        print("  report                                  Generate research report")
        print("  validate [phase]                        Check if phase can run")
        print("  reset <phase>                           Reset a phase to re-run it")
        print("  skip <phase>                            Skip a phase")
        print("  research                                Current research phase")
        print("  knowledge [--domain x] [--bottleneck y] Search queries + knowledge status")
        print("  techniques                              Show discovered techniques")
        print("  architect                               Architecture plan")
        print("  bootstrap                               Create project structure")
        print("  curriculum [domain]                     Show/create curriculum")
        print("  mutations                               List mutation types")
        print("  discover                                Feature discovery prompts")
        print()
        print("Phases: research, architect, bootstrap, build, test, optimize, report")
        print("Domains: web_api, ml_training, game, library, optimization, custom")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "status": cmd_status()
    elif cmd == "next": cmd_next()
    elif cmd == "run":
        max_exp = 200
        for i, arg in enumerate(sys.argv[2:], 2):
            if arg == "--max-experiments" and i + 1 < len(sys.argv):
                max_exp = int(sys.argv[i + 1])
        cmd_run(max_exp)
    elif cmd == "run-phase":
        cmd_run_phase(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "report": cmd_report()
    elif cmd == "validate":
        cmd_validate(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "reset":
        if len(sys.argv) < 3:
            print("Usage: reset <phase>")
            sys.exit(1)
        cmd_reset(sys.argv[2])
    elif cmd == "skip":
        if len(sys.argv) < 3:
            print("Usage: skip <phase>")
            sys.exit(1)
        cmd_skip(sys.argv[2])
    elif cmd == "research": cmd_research()
    elif cmd == "architect": cmd_architect()
    elif cmd == "bootstrap": cmd_bootstrap()
    elif cmd == "curriculum": cmd_curriculum(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "mutations": cmd_mutations()
    elif cmd == "discover": cmd_discover()
    elif cmd == "knowledge":
        kw = {}
        for i, arg in enumerate(sys.argv[2:], 2):
            if arg == "--domain" and i+1 < len(sys.argv): kw["domain"] = sys.argv[i+1]
            elif arg == "--spec" and i+1 < len(sys.argv): kw["spec"] = sys.argv[i+1]
            elif arg == "--bottleneck" and i+1 < len(sys.argv): kw["bottleneck"] = sys.argv[i+1]
        cmd_knowledge(**kw)
    elif cmd == "techniques": cmd_techniques()
    elif cmd == "init":
        spec = ""
        domain = "custom"
        for i, arg in enumerate(sys.argv[2:], 2):
            if arg == "--spec" and i+1 < len(sys.argv): spec = sys.argv[i+1]
            elif arg == "--domain" and i+1 < len(sys.argv): domain = sys.argv[i+1]
        cmd_init(spec, domain)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
