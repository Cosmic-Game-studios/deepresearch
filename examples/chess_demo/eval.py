#!/usr/bin/env python3
"""
Evaluation harness for DeepResearch.

Runs the chess engine tournament and outputs metrics in the format
DeepResearch expects: "metric_name value" per line.

Usage:
    python eval.py           # Full eval (50 games)
    python eval.py --quick   # Quick eval (10 games)
"""
import sys
import time

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from engine import run_tournament

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_games = 10 if quick else 50

    start = time.time()

    # Run tests first
    from tests.test_engine import (
        test_initial_board, test_legal_moves_initial, test_evaluate_initial,
        test_engine_finds_capture, test_checkmate_detection,
        test_game_completes, test_make_move_changes_turn,
    )
    tests = [
        test_initial_board, test_legal_moves_initial, test_evaluate_initial,
        test_engine_finds_capture, test_checkmate_detection,
        test_game_completes, test_make_move_changes_turn,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception:
            pass
    test_pass_rate = passed / len(tests)

    # Run tournament
    win_rate = run_tournament(n_games=n_games, verbose=False)

    elapsed = time.time() - start

    # Output metrics (DeepResearch format)
    print(f"test_pass_rate {test_pass_rate:.2f}")
    print(f"win_rate {win_rate:.3f}")
    print(f"eval_time_seconds {elapsed:.1f}")
    print(f"games_played {n_games}")
