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
from engine.autonomous import Orchestrator, DomainResearcher, Architect, Bootstrapper

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("DeepResearch Level 2-3 Engine")
        print()
        print("Commands:")
        print("  init [--spec 'text'] [--domain name]   Initialize Level 3 project")
        print("  status                                  Full pipeline status")
        print("  next                                    What should the agent do next?")
        print("  research                                Current research phase")
        print("  architect                               Architecture plan")
        print("  bootstrap                               Create project structure")
        print("  curriculum [domain]                     Show/create curriculum")
        print("  mutations                               List mutation types")
        print("  discover                                Feature discovery prompts")
        print()
        print("Domains: web_api, ml_training, game, library, optimization, custom")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "status": cmd_status()
    elif cmd == "next": cmd_next()
    elif cmd == "research": cmd_research()
    elif cmd == "architect": cmd_architect()
    elif cmd == "bootstrap": cmd_bootstrap()
    elif cmd == "curriculum": cmd_curriculum(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "mutations": cmd_mutations()
    elif cmd == "discover": cmd_discover()
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
