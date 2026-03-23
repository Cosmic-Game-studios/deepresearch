#!/usr/bin/env python3
"""
DeepResearch Integration Test — Verifies the complete Level 1-3 pipeline.

Tests:
1. Mutation system: propose, safety check, execute, rollback
2. Curriculum: create, track metrics, advance stages, detect regression
3. Pipeline: next_experiment, evaluate_and_decide, bandit updates
4. Orchestrator: phase progression, research, architecture, bootstrap
5. Feature discovery: pattern analysis

Run: python test_integration.py
"""

import json
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Ensure we can import engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


def check(condition, msg=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        return True
    else:
        FAIL += 1
        print(f"    ✗ FAILED: {msg}")
        return False


@test("Mutation types are defined correctly")
def test_mutation_types():
    from engine.mutations import MUTATION_TYPES
    check(len(MUTATION_TYPES) >= 6, "Need at least 6 mutation types")
    check("parametric" in MUTATION_TYPES, "Missing parametric")
    check("structural_addition" in MUTATION_TYPES, "Missing structural_addition")
    check("architectural" in MUTATION_TYPES, "Missing architectural")
    check(MUTATION_TYPES["parametric"]["level"] == 1, "parametric should be L1")
    check(MUTATION_TYPES["structural_addition"]["level"] == 2, "structural_addition should be L2")
    check(MUTATION_TYPES["architectural"]["level"] == 3, "architectural should be L3")
    check(MUTATION_TYPES["structural_addition"]["requires_tests"], "L2 should require tests")
    check(not MUTATION_TYPES["parametric"]["requires_tests"], "L1 shouldn't require tests")


@test("MutationManager propose and safety check")
def test_mutation_manager():
    from engine.mutations import MutationManager
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()
    (dr / "config.json").write_text(json.dumps({
        "target_files": ["src/app.py"],
        "read_only_files": ["tests/"],
        "mutation_levels": [1, 2],
        "test_command": "echo ok",
    }))

    mm = MutationManager(tmpdir)

    # Propose a mutation
    p = mm.propose("structural_addition", ["src/app.py"],
                    "Add caching", "DB is the bottleneck")
    check(p.id.startswith("exp-"), f"ID should start with exp-, got {p.id}")
    check(p.mutation_type == "structural_addition", "Wrong mutation type")

    # Safety check — allowed file
    safety = mm.check_safety(p)
    check(safety["safe"], f"Should be safe: {safety['reasons']}")

    # Safety check — read-only file
    p_bad = mm.propose("parametric", ["tests/test.py"],
                        "Change test", "testing")
    safety_bad = mm.check_safety(p_bad)
    check(not safety_bad["safe"], "Should NOT be safe for read-only file")

    shutil.rmtree(tmpdir)


@test("Snapshot and rollback")
def test_rollback():
    from engine.mutations import MutationManager
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()
    (dr / "config.json").write_text('{}')

    # Create a file
    src = Path(tmpdir) / "test.txt"
    src.write_text("original content")

    mm = MutationManager(tmpdir)
    snapshot = mm.snapshot_files(["test.txt"])
    check(snapshot["test.txt"] == "original content", "Snapshot should capture content")

    # Modify file
    src.write_text("modified content")
    check(src.read_text() == "modified content", "File should be modified")

    # Rollback
    mm.rollback(snapshot)
    check(src.read_text() == "original content", "Rollback should restore original")

    shutil.rmtree(tmpdir)


@test("Curriculum creation from templates")
def test_curriculum_templates():
    from engine.curriculum import CurriculumRunner
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()

    for domain in ["web_api", "ml_training", "game", "library", "optimization", "custom"]:
        runner = CurriculumRunner.create_from_template(domain, dr / "curriculum.json")
        check(len(runner.stages) > 0, f"Template '{domain}' should have stages")
        for s in runner.stages:
            check("name" in s, f"Stage missing name in {domain}")
            check("metric" in s, f"Stage missing metric in {domain}")
            check("target" in s, f"Stage missing target in {domain}")

    shutil.rmtree(tmpdir)


@test("Curriculum stage advancement")
def test_curriculum_advancement():
    from engine.curriculum import CurriculumRunner
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()

    runner = CurriculumRunner.create_from_template("web_api", dr / "curriculum.json")
    runner.history_path = dr / "curriculum_history_test.jsonl"

    # Should start at stage 1
    stage = runner.current_stage()
    check(stage is not None, "Should have a current stage")
    check(stage["name"] == "Correctness", f"Should start at Correctness, got {stage['name']}")

    # Update metrics — not enough to advance
    runner.update_metrics({"test_pass_rate": 0.5})
    check(not runner.check_advancement(), "Should NOT advance at 50%")

    # Update metrics — enough to advance
    runner.update_metrics({"test_pass_rate": 1.0})
    check(runner.check_advancement(), "Should advance at 100%")

    # Should now be at stage 2
    stage = runner.current_stage()
    check(stage["name"] == "Performance", f"Should be at Performance, got {stage['name']}")

    shutil.rmtree(tmpdir)


@test("Curriculum regression detection")
def test_curriculum_regression():
    from engine.curriculum import CurriculumRunner
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()

    runner = CurriculumRunner.create_from_template("web_api", dr / "curriculum.json")

    # Complete stage 1
    runner.update_metrics({"test_pass_rate": 1.0})
    runner.check_advancement()

    # Check regression
    regressions = runner.check_regression({"test_pass_rate": 0.8})
    check(len(regressions) > 0, "Should detect regression on Correctness")
    check("Correctness" in regressions, f"Should name the regressed stage, got {regressions}")

    shutil.rmtree(tmpdir)


@test("Pipeline next_experiment")
def test_pipeline_next():
    from engine.pipeline import ExperimentPipeline
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()
    (dr / "config.json").write_text(json.dumps({
        "target_files": ["src/"], "mutation_levels": [1, 2],
        "metric": "test_pass_rate", "metric_direction": "higher",
        "experiment_budget": 100,
    }))
    (dr / "strategy-state.json").write_text(json.dumps({
        "temperature": 1.0, "total_experiments": 0, "bandit_arms": {},
        "no_improvement_streak": 0, "best_metric": None,
    }))
    (dr / "curriculum.json").write_text(json.dumps({"stages": [
        {"name": "Test", "metric": "test_pass_rate", "target": 1.0, "direction": "higher"}
    ]}))

    # Change to tmpdir so engine finds .deepresearch
    old_cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        pipe = ExperimentPipeline(tmpdir)
        inst = pipe.next_experiment()
        check("experiment_id" in inst, "Should have experiment_id")
        check("instructions" in inst, "Should have instructions")
        check("temperature" in inst, "Should have temperature")
        check(inst["experiment_id"] == "exp-0001", f"First exp should be 0001, got {inst['experiment_id']}")
    finally:
        os.chdir(old_cwd)

    shutil.rmtree(tmpdir)


@test("Orchestrator phase progression")
def test_orchestrator():
    from engine.autonomous import Orchestrator
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()

    old_cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        orch = Orchestrator(spec="Build a calculator")
        orch.save_state()

        # Should start in research phase
        check(orch.current_phase == "research", f"Should start in research, got {orch.current_phase}")

        action = orch.get_next_action()
        check(action["action"] == "research", f"First action should be research, got {action['action']}")

        # Advance through phases
        orch.advance_phase()
        check(orch.current_phase == "architect", f"Should be architect, got {orch.current_phase}")

        orch.advance_phase()
        check(orch.current_phase == "bootstrap", f"Should be bootstrap, got {orch.current_phase}")

        orch.advance_phase()
        check(orch.current_phase == "build", f"Should be build, got {orch.current_phase}")

        orch.advance_phase()
        check(orch.current_phase == "test", f"Should be test, got {orch.current_phase}")

        orch.advance_phase()
        check(orch.current_phase == "optimize", f"Should be optimize, got {orch.current_phase}")

        orch.advance_phase()
        check(orch.current_phase == "report", f"Should be report, got {orch.current_phase}")

        orch.advance_phase()
        check(orch.current_phase == "complete", f"Should be complete, got {orch.current_phase}")
    finally:
        os.chdir(old_cwd)

    shutil.rmtree(tmpdir)


@test("Architect component ordering")
def test_architect():
    from engine.autonomous import Architect, Component
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()

    old_cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        arch = Architect()
        arch.add_component("database", "Data layer", files=["src/db.py"])
        arch.add_component("api", "REST endpoints", depends_on=["database"], files=["src/api.py"])
        arch.add_component("auth", "Authentication", depends_on=["database"], files=["src/auth.py"])
        arch.add_component("frontend", "UI", depends_on=["api", "auth"], files=["src/ui.py"])

        order = arch.get_build_order()
        check(order[0] == "database", f"Database should be first, got {order[0]}")
        check(order[-1] == "frontend", f"Frontend should be last, got {order[-1]}")
        check(order.index("api") > order.index("database"), "API after database")
        check(order.index("frontend") > order.index("api"), "Frontend after API")

        # Next component should be database (first unbuilt)
        nxt = arch.next_component()
        check(nxt.name == "database", f"Next should be database, got {nxt.name}")

        arch.update_status("database", "tested")
        nxt = arch.next_component()
        check(nxt.name in ("api", "auth"), f"Next should be api or auth, got {nxt.name}")
    finally:
        os.chdir(old_cwd)

    shutil.rmtree(tmpdir)


@test("Bootstrapper creates project structure")
def test_bootstrapper():
    from engine.autonomous import Architect, Bootstrapper
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()

    old_cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        arch = Architect()
        arch.add_component("core", "Core logic", files=["src/core.py"], test_file="tests/test_core.py")
        arch.add_component("utils", "Utilities", files=["src/utils.py"], test_file="tests/test_utils.py")

        created = Bootstrapper.bootstrap(arch, tmpdir)
        check(len(created) > 0, "Should create files")
        check(Path(tmpdir, "src/core.py").exists(), "core.py should exist")
        check(Path(tmpdir, "tests/test_core.py").exists(), "test_core.py should exist")
        check(Path(tmpdir, ".deepresearch/config.json").exists(), "config.json should exist")

        # Verify content
        core_content = Path(tmpdir, "src/core.py").read_text()
        check("Core logic" in core_content, "Stub should mention purpose")
    finally:
        os.chdir(old_cwd)

    shutil.rmtree(tmpdir)


@test("FeatureDiscovery generates analysis prompt")
def test_feature_discovery():
    from engine.mutations import FeatureDiscovery
    prompt = FeatureDiscovery.generate_analysis_prompt()
    check("caching" in prompt, "Should mention caching pattern")
    check("error_handling" in prompt, "Should mention error handling")
    check("Performance" in prompt, "Should have Performance category")
    check("Reliability" in prompt, "Should have Reliability category")

    patterns = FeatureDiscovery.suggest_analysis("performance")
    check(len(patterns) >= 4, f"Performance should have 4+ patterns, got {len(patterns)}")


@test("Curriculum progress report formatting")
def test_curriculum_report():
    from engine.curriculum import CurriculumRunner
    tmpdir = tempfile.mkdtemp()
    dr = Path(tmpdir) / ".deepresearch"
    dr.mkdir()

    runner = CurriculumRunner.create_from_template("game", dr / "curriculum.json")
    runner.update_metrics({"no_crash_rate": 0.85})
    report = runner.progress_report()
    check("Curriculum Progress" in report, "Should have header")
    check("← CURRENT" in report, "Should show current stage")
    check("locked" in report, "Should show locked stages")

    shutil.rmtree(tmpdir)


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'═'*60}")
    print(f"  DeepResearch Integration Tests")
    print(f"{'═'*60}\n")

    for name, fn in TESTS:
        print(f"  ▶ {name}")
        try:
            fn()
            print(f"    ✓ passed")
        except Exception as e:
            FAIL += 1
            print(f"    ✗ EXCEPTION: {e}")

    print(f"\n{'═'*60}")
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} checks passed, {FAIL} failed")
    if FAIL == 0:
        print(f"  ✅ ALL TESTS PASS")
    else:
        print(f"  ❌ {FAIL} FAILURES")
    print(f"{'═'*60}")
    sys.exit(1 if FAIL > 0 else 0)
