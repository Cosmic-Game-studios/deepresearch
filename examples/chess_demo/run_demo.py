#!/usr/bin/env python3
"""
DeepResearch Demo — Chess Engine across all 3 Levels

This demo shows how DeepResearch optimizes a chess engine at each level:

  Level 1: Tune piece values and search depth (parametric mutations)
  Level 2: Add features like move ordering (structural mutations)
  Level 3: Full pipeline from specification (autonomous engineer)

Run: python run_demo.py
"""
import sys
import os
import json
import time
from pathlib import Path

# Setup paths — import chess engine via importlib to avoid name collision with deepresearch engine/
DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = str(DEMO_DIR.parent.parent)

import importlib.util
_spec = importlib.util.spec_from_file_location("chess_engine", str(DEMO_DIR / "engine.py"))
chess_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chess_engine)

Board = chess_engine.Board
evaluate = chess_engine.evaluate
find_best_move = chess_engine.find_best_move
run_tournament = chess_engine.run_tournament
PIECE_VALUES = chess_engine.PIECE_VALUES
SEARCH_DEPTH = chess_engine.SEARCH_DEPTH
PAWN_PST = chess_engine.PAWN_PST
KNIGHT_PST = chess_engine.KNIGHT_PST

sys.path.insert(0, PROJECT_ROOT)


def banner(text):
    print(f"\n{'═'*60}")
    print(f"  {text}")
    print(f"{'═'*60}\n")


def run_baseline():
    """Establish baseline win rate."""
    banner("BASELINE — Current engine vs random")
    win_rate = run_tournament(n_games=50)
    return win_rate


# ════════════════════════════════════════════════════════════
# LEVEL 1 — Parameter Tuning
# ════════════════════════════════════════════════════════════

def demo_level1(baseline):
    """
    Level 1: Tune existing parameters.

    The DeepResearch loop:
    1. R1 DEEP READ: Read engine.py, identify tunable parameters
    2. R2 HYPOTHESIZE: "Knight is undervalued at 300, should be 320"
    3. R3 PREDICT: "Expect +2% win rate improvement"
    4. MUTATE: Change PIECE_VALUES['N'] from 300 to 320
    5. EVALUATE: Run tournament
    6. REFLECT: Compare prediction to actual result
    """
    banner("LEVEL 1 — Parameter Tuning")
    print("Tunable parameters:")
    print(f"  Piece values: P={chess_engine.PIECE_VALUES['P']}, N={chess_engine.PIECE_VALUES['N']}, "
          f"B={chess_engine.PIECE_VALUES['B']}, R={chess_engine.PIECE_VALUES['R']}, Q={chess_engine.PIECE_VALUES['Q']}")
    print(f"  Search depth: {chess_engine.SEARCH_DEPTH}")
    print()

    experiments = []

    # Experiment 1: Increase knight value (common chess knowledge: knights are worth ~3.2 pawns)
    print("Experiment L1-001: Increase knight value 300 → 320")
    print("  Hypothesis: Knights are slightly undervalued, better evaluation should improve play")
    original_n = chess_engine.PIECE_VALUES['N']
    chess_engine.PIECE_VALUES['N'] = 320
    chess_engine.PIECE_VALUES['n'] = -320
    wr = run_tournament(n_games=30, verbose=False)
    improvement = (wr - baseline) / max(baseline, 0.01) * 100
    kept = improvement > -3  # keep unless significantly worse
    if not kept:
        chess_engine.PIECE_VALUES['N'] = original_n
        chess_engine.PIECE_VALUES['n'] = -original_n
    experiments.append({"id": "L1-001", "change": "N 300→320", "win_rate": wr,
                        "improvement": improvement, "status": "kept" if kept else "reverted"})
    print(f"  Result: win_rate={wr:.3f} ({improvement:+.1f}%) → {'KEPT' if kept else 'REVERTED'}")

    # Experiment 2: Increase bishop value (bishop pair is valuable)
    print("\nExperiment L1-002: Increase bishop value 310 → 330")
    print("  Hypothesis: Bishop pair advantage means bishops should be worth more")
    original_b = chess_engine.PIECE_VALUES['B']
    chess_engine.PIECE_VALUES['B'] = 330
    chess_engine.PIECE_VALUES['b'] = -330
    wr = run_tournament(n_games=30, verbose=False)
    improvement = (wr - baseline) / max(baseline, 0.01) * 100
    kept = improvement > -3
    if not kept:
        chess_engine.PIECE_VALUES['B'] = original_b
        chess_engine.PIECE_VALUES['b'] = -original_b
    experiments.append({"id": "L1-002", "change": "B 310→330", "win_rate": wr,
                        "improvement": improvement, "status": "kept" if kept else "reverted"})
    print(f"  Result: win_rate={wr:.3f} ({improvement:+.1f}%) → {'KEPT' if kept else 'REVERTED'}")

    # Summary
    print(f"\nLevel 1 summary: {sum(1 for e in experiments if e['status']=='kept')}/{len(experiments)} kept")
    return experiments


# ════════════════════════════════════════════════════════════
# LEVEL 2 — Feature Addition
# ════════════════════════════════════════════════════════════

def demo_level2(baseline):
    """
    Level 2: Add new code to the engine.

    The DeepResearch loop:
    1. R1 DEEP READ: The engine has no move ordering → alpha-beta wastes time
    2. R2 HYPOTHESIZE: "Adding MVV-LVA move ordering will improve pruning"
    3. R3 PREDICT: "Expect +5-10% win rate from better search efficiency"
    4. MUTATE: Set USE_MOVE_ORDERING = True (structural_addition — the ordering code exists)
    5. EVALUATE: Run tournament
    6. REFLECT: Did move ordering help? Why/why not?
    """
    banner("LEVEL 2 — Feature Addition")
    experiments = []

    # Experiment 1: Enable move ordering
    print("Experiment L2-001: Enable move ordering (MVV-LVA)")
    print("  Hypothesis: Ordering captures first makes alpha-beta prune more efficiently")
    print("  Mutation type: structural_addition (the code exists but is disabled)")
    chess_engine.USE_MOVE_ORDERING = True
    wr = run_tournament(n_games=30, verbose=False)
    improvement = (wr - baseline) / max(baseline, 0.01) * 100
    kept = improvement > -3
    if not kept:
        chess_engine.USE_MOVE_ORDERING = False
    experiments.append({"id": "L2-001", "change": "move_ordering=True", "win_rate": wr,
                        "improvement": improvement, "status": "kept" if kept else "reverted"})
    print(f"  Result: win_rate={wr:.3f} ({improvement:+.1f}%) → {'KEPT' if kept else 'REVERTED'}")

    # Experiment 2: Enable transposition table
    print("\nExperiment L2-002: Enable transposition table")
    print("  Hypothesis: TT avoids re-searching identical positions")
    print("  Mutation type: structural_addition")
    chess_engine.USE_TT = True
    wr = run_tournament(n_games=30, verbose=False)
    improvement = (wr - baseline) / max(baseline, 0.01) * 100
    kept = improvement > -3
    if not kept:
        chess_engine.USE_TT = False
    experiments.append({"id": "L2-002", "change": "transposition_table=True", "win_rate": wr,
                        "improvement": improvement, "status": "kept" if kept else "reverted"})
    print(f"  Result: win_rate={wr:.3f} ({improvement:+.1f}%) → {'KEPT' if kept else 'REVERTED'}")

    print(f"\nLevel 2 summary: {sum(1 for e in experiments if e['status']=='kept')}/{len(experiments)} kept")
    return experiments


# ════════════════════════════════════════════════════════════
# LEVEL 3 — Full Pipeline Demo
# ════════════════════════════════════════════════════════════

def demo_level3():
    """
    Level 3: Show the full autonomous pipeline.

    This demonstrates the Orchestrator driving through all 7 phases.
    In a real run, the LLM agent would be writing the code.
    Here we show the pipeline mechanics.
    """
    banner("LEVEL 3 — Autonomous Pipeline Demo")

    from engine.autonomous import Orchestrator, Bootstrapper

    # Initialize
    spec = (
        "Build a chess engine that beats random players 90%+ of the time. "
        "Must include: board representation, move generation, evaluation, "
        "alpha-beta search, and tournament evaluation harness."
    )

    # Use temp directory to avoid clobbering
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = Orchestrator(spec=spec, project_root=tmpdir)
        orch.save_state()

        print(f"Spec: {spec[:80]}...")
        print(f"Project: {tmpdir}")
        print()

        # Phase 0: Research
        print("Phase 0: RESEARCH")
        result = orch.run()
        assert result['status'] == 'needs_agent'
        print(f"  Status: needs_agent (research phase: {result['current_action'].get('research_phase', '?')})")
        # Simulate agent completing all research
        orch.researcher.complete_phase('understand_spec', {
            'input': 'Chess position (FEN or board array)',
            'output': 'Best move + evaluation score',
            'constraints': 'Pure Python, no external libraries',
            'primary_metric': 'win_rate vs random player',
        })
        orch.researcher.complete_phase('survey_existing', {
            'standard_approach': 'Alpha-beta with iterative deepening',
            'key_algorithms': 'Negamax, MVV-LVA, Transposition tables',
            'pitfalls': 'Slow move generation, no quiescence = horizon effect',
        })
        orch.researcher.complete_phase('identify_architecture', {
            'components': 'Board, MoveGen, Eval, Search, Tournament',
            'data_flow': 'Board → MoveGen → Search(Eval) → BestMove',
        })
        orch.researcher.complete_phase('plan_testing', {
            'correctness': 'Known positions, perft counts',
            'performance': 'Win rate vs random (target: 90%+)',
        })
        print("  Simulated: all 4 research phases completed")

        # Phase 1: Architect
        result = orch.run()
        print(f"\nPhase 1: ARCHITECT")
        print(f"  Status: {result['status']} (research auto-completed)")
        # Simulate agent designing architecture
        orch.architect.add_component('board', 'Board representation and move generation',
                                     files=['src/board.py'], test_file='tests/test_board.py',
                                     estimated_experiments=3)
        orch.architect.add_component('eval', 'Position evaluation with material + PST',
                                     files=['src/eval.py'], depends_on=['board'],
                                     test_file='tests/test_eval.py', estimated_experiments=5)
        orch.architect.add_component('search', 'Alpha-beta search with move ordering',
                                     files=['src/search.py'], depends_on=['board', 'eval'],
                                     test_file='tests/test_search.py', estimated_experiments=5)
        orch.architect.save()
        print(f"  Components: {[c.name for c in orch.architect.components]}")
        print(f"  Build order: {orch.architect.get_build_order()}")

        # Phase 2: Bootstrap (auto-runs)
        result = orch.run()
        print(f"\nPhase 2: BOOTSTRAP")
        auto_phases = result.get('phases_completed', [])
        print(f"  Auto-completed phases: {auto_phases}")

        # Phase 3: Build
        print(f"\nPhase 3: BUILD")
        print(f"  Status: {result['status']}")
        if result.get('current_action'):
            comp = result['current_action'].get('component', {})
            print(f"  Next component: {comp.get('name', '?')}")
            print(f"  Mutation type: {result['current_action'].get('mutation_type', '?')}")
            if result['current_action'].get('pipeline'):
                pipe = result['current_action']['pipeline']
                print(f"  Pipeline: T={pipe.get('temperature', '?')}, experiments={pipe.get('experiment_count', 0)}")

        # Simulate building all components
        for comp in orch.architect.components:
            orch.architect.update_status(comp.name, 'tested')
            orch.record_experiment('build', f'Implemented {comp.name}')

        # Continue to test/optimize
        result = orch.run()
        print(f"\nPhase 4: TEST")
        print(f"  Status: {result['status']}")
        print(f"  (Would run test suite and fix failures)")

        # Generate report
        print(f"\nPhase 6: REPORT")
        report = orch.generate_report()
        print(f"  Report length: {len(report)} chars")
        print(f"  Total experiments tracked: {orch.state.get('total_experiments', 0)}")

        # Show pipeline status
        print()
        print(orch.status_report())


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    banner("DeepResearch Demo — Chess Engine (All 3 Levels)")

    print("This demo shows DeepResearch optimizing a chess engine:")
    print("  Level 1: Tune piece values (parametric mutations)")
    print("  Level 2: Add move ordering + TT (structural mutations)")
    print("  Level 3: Full autonomous pipeline from specification")
    print()

    # Baseline
    baseline = run_baseline()
    print(f"\nBaseline win rate: {baseline:.3f}")

    # Level 1
    l1_results = demo_level1(baseline)

    # Update baseline after L1
    new_baseline = run_tournament(n_games=30, verbose=False)
    print(f"\nWin rate after L1: {new_baseline:.3f}")

    # Level 2
    l2_results = demo_level2(new_baseline)

    # Final measurement
    final = run_tournament(n_games=50, verbose=False)
    print(f"\nWin rate after L2: {final:.3f}")

    # Level 3
    demo_level3()

    # Summary
    banner("DEMO COMPLETE — Summary")
    print(f"  Baseline:   {baseline:.3f}")
    print(f"  After L1:   {new_baseline:.3f} ({(new_baseline-baseline)/max(baseline,0.01)*100:+.1f}%)")
    print(f"  After L2:   {final:.3f} ({(final-baseline)/max(baseline,0.01)*100:+.1f}%)")
    print()
    print("  L1 experiments: " + ", ".join(
        f"{e['id']}={e['status']}" for e in l1_results))
    print("  L2 experiments: " + ", ".join(
        f"{e['id']}={e['status']}" for e in l2_results))
    print()
    print("  This is what DeepResearch does automatically:")
    print("  Read → Hypothesize → Predict → Mutate → Evaluate → Reflect")
    print("  The same loop, at every level, on any problem with a metric.")


if __name__ == "__main__":
    main()
