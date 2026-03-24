"""
Mini Chess Engine — DeepResearch Demo

A minimal but functional chess engine designed to be optimized
by DeepResearch across all 3 levels:

  Level 1: Tune piece values, PST weights, search depth
  Level 2: Add move ordering, quiescence search, transposition table
  Level 3: Built from spec using the full pipeline

The engine is intentionally simple so improvements are measurable.
Metric: win rate against random player (higher = better).
"""

import random
import time

# ════════════════════════════════════════════════════════════
# PIECE VALUES — Level 1 tuning target
# ════════════════════════════════════════════════════════════

PIECE_VALUES = {
    'P': 100, 'N': 300, 'B': 310, 'R': 500, 'Q': 900, 'K': 0,
    'p': -100, 'n': -300, 'b': -310, 'r': -500, 'q': -900, 'k': 0,
}

# Piece-square tables (white's perspective, index 0=a8, 63=h1)
# Level 1: tune these values
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST = {'P': PAWN_PST, 'N': KNIGHT_PST}

# ════════════════════════════════════════════════════════════
# SEARCH PARAMETERS — Level 1 tuning target
# ════════════════════════════════════════════════════════════

SEARCH_DEPTH = 3          # L1: tune 1-5
QUIESCENCE_DEPTH = 0      # L2: add quiescence search (set to 2+)
USE_MOVE_ORDERING = False  # L2: add move ordering
USE_TT = False             # L2: add transposition table

# ════════════════════════════════════════════════════════════
# BOARD REPRESENTATION
# ════════════════════════════════════════════════════════════

INITIAL_BOARD = [
    'r','n','b','q','k','b','n','r',
    'p','p','p','p','p','p','p','p',
    '.','.','.','.','.','.','.','.',
    '.','.','.','.','.','.','.','.',
    '.','.','.','.','.','.','.','.',
    '.','.','.','.','.','.','.','.',
    'P','P','P','P','P','P','P','P',
    'R','N','B','Q','K','B','N','R',
]

def is_white(p):
    return p.isupper()

def is_black(p):
    return p.islower()

def is_piece(p, color):
    return (p.isupper() if color == 'w' else p.islower()) and p != '.'


class Board:
    """Simple 8x8 array board."""

    def __init__(self):
        self.squares = list(INITIAL_BOARD)
        self.turn = 'w'  # 'w' or 'b'
        self.move_count = 0
        self.castling = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant = -1  # square index or -1
        self.history = []

    def copy(self):
        b = Board.__new__(Board)
        b.squares = list(self.squares)
        b.turn = self.turn
        b.move_count = self.move_count
        b.castling = dict(self.castling)
        b.en_passant = self.en_passant
        b.history = list(self.history)
        return b

    def get(self, sq):
        return self.squares[sq]

    def set(self, sq, piece):
        self.squares[sq] = piece

    def row(self, sq):
        return sq >> 3

    def col(self, sq):
        return sq & 7

    def sq(self, r, c):
        return (r << 3) | c

    def in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def find_king(self, color):
        target = 'K' if color == 'w' else 'k'
        for i in range(64):
            if self.squares[i] == target:
                return i
        return -1

    def is_attacked(self, sq, by_color):
        """Check if square is attacked by given color."""
        r, c = self.row(sq), self.col(sq)

        # Knight attacks
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r+dr, c+dc
            if self.in_bounds(nr, nc):
                p = self.get(self.sq(nr, nc))
                if p.upper() == 'N' and is_piece(p, by_color):
                    return True

        # King attacks
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if self.in_bounds(nr, nc):
                    p = self.get(self.sq(nr, nc))
                    if p.upper() == 'K' and is_piece(p, by_color):
                        return True

        # Pawn attacks
        pawn_dir = 1 if by_color == 'w' else -1  # direction pawns attack FROM
        for dc in [-1, 1]:
            nr, nc = r + pawn_dir, c + dc
            if self.in_bounds(nr, nc):
                p = self.get(self.sq(nr, nc))
                if p.upper() == 'P' and is_piece(p, by_color):
                    return True

        # Sliding: bishop/queen (diagonals)
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            while self.in_bounds(nr, nc):
                p = self.get(self.sq(nr, nc))
                if p != '.':
                    if is_piece(p, by_color) and p.upper() in ('B', 'Q'):
                        return True
                    break
                nr, nc = nr+dr, nc+dc

        # Sliding: rook/queen (straights)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            while self.in_bounds(nr, nc):
                p = self.get(self.sq(nr, nc))
                if p != '.':
                    if is_piece(p, by_color) and p.upper() in ('R', 'Q'):
                        return True
                    break
                nr, nc = nr+dr, nc+dc

        return False

    def in_check(self, color):
        king_sq = self.find_king(color)
        if king_sq < 0:
            return True
        opp = 'b' if color == 'w' else 'w'
        return self.is_attacked(king_sq, opp)

    def generate_moves(self):
        """Generate all pseudo-legal moves for current side."""
        moves = []
        color = self.turn

        for sq in range(64):
            p = self.get(sq)
            if not is_piece(p, color):
                continue

            r, c = self.row(sq), self.col(sq)
            pt = p.upper()

            if pt == 'P':
                direction = -1 if color == 'w' else 1
                # Forward
                nr = r + direction
                if self.in_bounds(nr, c) and self.get(self.sq(nr, c)) == '.':
                    if nr == 0 or nr == 7:
                        for promo in (['Q','R','B','N'] if color == 'w' else ['q','r','b','n']):
                            moves.append((sq, self.sq(nr, c), promo))
                    else:
                        moves.append((sq, self.sq(nr, c), None))
                    # Double push from start
                    start_row = 6 if color == 'w' else 1
                    if r == start_row:
                        nr2 = r + 2 * direction
                        if self.get(self.sq(nr2, c)) == '.':
                            moves.append((sq, self.sq(nr2, c), None))
                # Captures
                for dc in [-1, 1]:
                    nc = c + dc
                    if self.in_bounds(nr, nc):
                        target_sq = self.sq(nr, nc)
                        target = self.get(target_sq)
                        if (target != '.' and not is_piece(target, color)) or target_sq == self.en_passant:
                            if nr == 0 or nr == 7:
                                for promo in (['Q','R','B','N'] if color == 'w' else ['q','r','b','n']):
                                    moves.append((sq, target_sq, promo))
                            else:
                                moves.append((sq, target_sq, None))

            elif pt == 'N':
                for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                    nr, nc = r+dr, c+dc
                    if self.in_bounds(nr, nc):
                        target = self.get(self.sq(nr, nc))
                        if target == '.' or not is_piece(target, color):
                            moves.append((sq, self.sq(nr, nc), None))

            elif pt == 'K':
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r+dr, c+dc
                        if self.in_bounds(nr, nc):
                            target = self.get(self.sq(nr, nc))
                            if target == '.' or not is_piece(target, color):
                                moves.append((sq, self.sq(nr, nc), None))

            else:
                # Sliding pieces
                directions = []
                if pt in ('B', 'Q'):
                    directions += [(-1,-1),(-1,1),(1,-1),(1,1)]
                if pt in ('R', 'Q'):
                    directions += [(-1,0),(1,0),(0,-1),(0,1)]
                for dr, dc in directions:
                    nr, nc = r+dr, c+dc
                    while self.in_bounds(nr, nc):
                        target = self.get(self.sq(nr, nc))
                        if target == '.':
                            moves.append((sq, self.sq(nr, nc), None))
                        elif not is_piece(target, color):
                            moves.append((sq, self.sq(nr, nc), None))
                            break
                        else:
                            break
                        nr, nc = nr+dr, nc+dc

        return moves

    def legal_moves(self):
        """Generate legal moves (filter out moves that leave king in check)."""
        legal = []
        for move in self.generate_moves():
            b2 = self.copy()
            b2.make_move(move)
            # After making the move, check if OUR king is in check
            if not b2.in_check(self.turn):
                legal.append(move)
        return legal

    def make_move(self, move):
        """Apply a move. Returns captured piece."""
        frm, to, promo = move
        piece = self.get(frm)
        captured = self.get(to)

        # En passant capture
        if piece.upper() == 'P' and to == self.en_passant:
            ep_capture_sq = to + (8 if self.turn == 'w' else -8)
            captured = self.get(ep_capture_sq)
            self.set(ep_capture_sq, '.')

        # Update en passant
        self.en_passant = -1
        if piece.upper() == 'P' and abs(frm - to) == 16:
            self.en_passant = (frm + to) // 2

        # Move piece
        self.set(to, promo if promo else piece)
        self.set(frm, '.')

        # Update castling rights
        if piece == 'K':
            self.castling['K'] = False
            self.castling['Q'] = False
        elif piece == 'k':
            self.castling['k'] = False
            self.castling['q'] = False
        if frm == 63 or to == 63: self.castling['K'] = False
        if frm == 56 or to == 56: self.castling['Q'] = False
        if frm == 7 or to == 7: self.castling['k'] = False
        if frm == 0 or to == 0: self.castling['q'] = False

        self.history.append(move)
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.move_count += 1
        return captured


# ════════════════════════════════════════════════════════════
# EVALUATION — Level 1 and Level 2 target
# ════════════════════════════════════════════════════════════

def evaluate(board):
    """Static evaluation from white's perspective."""
    score = 0
    for sq in range(64):
        p = board.get(sq)
        if p == '.':
            continue
        # Material
        score += PIECE_VALUES.get(p, 0)
        # Piece-square tables
        pt = p.upper()
        if pt in PST:
            if p.isupper():
                score += PST[pt][sq]
            else:
                score -= PST[pt][63 - sq]
    return score


# ════════════════════════════════════════════════════════════
# SEARCH — Level 1 (depth), Level 2 (features)
# ════════════════════════════════════════════════════════════

def order_moves(board, moves):
    """Level 2 feature: order moves for better alpha-beta pruning."""
    if not USE_MOVE_ORDERING:
        return moves

    def score(move):
        _, to, promo = move
        s = 0
        captured = board.get(to)
        if captured != '.':
            s += abs(PIECE_VALUES.get(captured, 0)) * 10
        if promo:
            s += abs(PIECE_VALUES.get(promo.upper(), 0))
        return s

    return sorted(moves, key=score, reverse=True)


# Transposition table (Level 2)
_tt = {}

def negamax(board, depth, alpha, beta, color_sign):
    """Negamax with alpha-beta pruning."""
    # Check transposition table
    if USE_TT:
        key = (tuple(board.squares), board.turn, depth)
        if key in _tt:
            return _tt[key]

    if depth <= 0:
        return color_sign * evaluate(board), None

    moves = board.legal_moves()
    if not moves:
        if board.in_check(board.turn):
            return -99999 + board.move_count, None  # checkmate
        return 0, None  # stalemate

    moves = order_moves(board, moves)

    best_score = -100000
    best_move = moves[0]

    for move in moves:
        b2 = board.copy()
        b2.make_move(move)
        score, _ = negamax(b2, depth - 1, -beta, -alpha, -color_sign)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            break

    if USE_TT:
        _tt[key] = (best_score, best_move)

    return best_score, best_move


def find_best_move(board, depth=None):
    """Find the best move for the current side."""
    if depth is None:
        depth = SEARCH_DEPTH
    color_sign = 1 if board.turn == 'w' else -1
    _tt.clear()
    score, move = negamax(board, depth, -100000, 100000, color_sign)
    return move, score


# ════════════════════════════════════════════════════════════
# GAME PLAY
# ════════════════════════════════════════════════════════════

def play_game(white_fn, black_fn, max_moves=200, verbose=False):
    """
    Play a game between two move functions.
    Returns: 'w' (white wins), 'b' (black wins), 'd' (draw)
    """
    board = Board()

    for _ in range(max_moves):
        moves = board.legal_moves()
        if not moves:
            if board.in_check(board.turn):
                return 'b' if board.turn == 'w' else 'w'
            return 'd'

        if board.turn == 'w':
            move = white_fn(board, moves)
        else:
            move = black_fn(board, moves)

        if move not in moves:
            move = moves[0]  # fallback

        board.make_move(move)

    return 'd'  # max moves reached


def random_player(board, moves):
    """Random move selection."""
    return random.choice(moves)


def engine_player(board, moves):
    """Engine move selection using search."""
    move, _ = find_best_move(board)
    if move is None or move not in moves:
        return moves[0]
    return move


def run_tournament(n_games=100, verbose=True):
    """
    Run engine vs random tournament.
    Returns win rate (0-1), which is the primary metric.
    """
    results = {'w': 0, 'b': 0, 'd': 0}
    start = time.time()

    for i in range(n_games):
        # Alternate colors
        if i % 2 == 0:
            result = play_game(engine_player, random_player, max_moves=150)
            if result == 'w':
                results['w'] += 1
            elif result == 'b':
                results['b'] += 1
            else:
                results['d'] += 1
        else:
            result = play_game(random_player, engine_player, max_moves=150)
            if result == 'b':
                results['w'] += 1
            elif result == 'w':
                results['b'] += 1
            else:
                results['d'] += 1

    elapsed = time.time() - start
    total = n_games
    win_rate = results['w'] / total

    if verbose:
        print(f"Tournament: {n_games} games in {elapsed:.1f}s")
        print(f"  Engine wins: {results['w']} ({results['w']/total*100:.0f}%)")
        print(f"  Random wins: {results['b']} ({results['b']/total*100:.0f}%)")
        print(f"  Draws:       {results['d']} ({results['d']/total*100:.0f}%)")
        print(f"  Win rate:    {win_rate:.3f}")

    return win_rate


if __name__ == "__main__":
    print("Chess Engine — DeepResearch Demo")
    print("Running 50-game tournament vs random player...")
    print()
    win_rate = run_tournament(n_games=50)
    print(f"\nwin_rate {win_rate:.3f}")
