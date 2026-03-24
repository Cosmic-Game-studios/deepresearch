"""Tests for chess engine."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from engine import Board, evaluate, find_best_move, random_player, play_game


def test_initial_board():
    """Board initializes correctly."""
    b = Board()
    assert b.get(0) == 'r'   # black rook a8
    assert b.get(4) == 'k'   # black king e8
    assert b.get(60) == 'K'  # white king e1
    assert b.get(63) == 'R'  # white rook h1
    assert b.turn == 'w'


def test_legal_moves_initial():
    """White has 20 legal moves in starting position."""
    b = Board()
    moves = b.legal_moves()
    assert len(moves) == 20, f"Expected 20 moves, got {len(moves)}"


def test_evaluate_initial():
    """Initial position should be roughly equal."""
    b = Board()
    score = evaluate(b)
    assert -50 < score < 50, f"Initial eval should be ~0, got {score}"


def test_engine_finds_capture():
    """Engine should capture a free queen."""
    b = Board()
    # Clear board, put white king and knight, black queen in knight's range
    b.squares = ['.'] * 64
    b.squares[60] = 'K'  # white king e1
    b.squares[4] = 'k'   # black king e8
    b.squares[36] = 'N'  # white knight e5
    b.squares[21] = 'q'  # black queen f7 — in knight range from e5
    b.turn = 'w'
    move, score = find_best_move(b, depth=2)
    assert move is not None
    _, to, _ = move
    assert to == 21, f"Expected capture on f7 (21), got {to}"


def test_checkmate_detection():
    """Engine detects checkmate (ladder mate: two rooks on 7th and 8th rank)."""
    b = Board()
    b.squares = ['.'] * 64
    # Ladder mate: king on a8, rooks on a1 and b7 would be complex
    # Simpler: king on h8, queen on g7 protected by bishop
    b.squares[7] = 'k'    # black king h8
    b.squares[14] = 'Q'   # white queen g7 (check + covers h8,g8,f8,h7,g6)
    b.squares[21] = 'B'   # white bishop f6 (protects queen on g7)
    b.squares[56] = 'K'   # white king a1
    b.turn = 'b'
    assert b.in_check('b'), "Black should be in check"
    moves = b.legal_moves()
    assert len(moves) == 0, f"Should be checkmate, got {len(moves)} moves"


def test_game_completes():
    """A game between two random players completes."""
    result = play_game(random_player, random_player, max_moves=100)
    assert result in ('w', 'b', 'd')


def test_make_move_changes_turn():
    """Making a move switches the turn."""
    b = Board()
    assert b.turn == 'w'
    moves = b.legal_moves()
    b.make_move(moves[0])
    assert b.turn == 'b'


if __name__ == "__main__":
    tests = [
        test_initial_board,
        test_legal_moves_initial,
        test_evaluate_initial,
        test_engine_finds_capture,
        test_checkmate_detection,
        test_game_completes,
        test_make_move_changes_turn,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    test_pass_rate = passed / len(tests)
    print(f"test_pass_rate {test_pass_rate:.2f}")
