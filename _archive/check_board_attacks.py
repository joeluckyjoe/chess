# File: check_board_attacks.py (save this in your ~/chess folder)
import chess

# Test for board.attacks() Omitting King's Square
# FEN where white rook on d5 attacks black king on d8.
fen_attacks = "3k4/3p4/5n2/3R2B1/4P3/8/8/K7 w - - 0 1"
board_attacks = chess.Board(fen_attacks)

print("--- Testing board.attacks() Omitting King's Square ---")
print(f"FEN: {board_attacks.fen()}")
# White rook is on D5 (square index 27)
# Black king is on D8 (square index 59)
rook_square = chess.D5
king_square = chess.D8

rook_attack_set = board_attacks.attacks(rook_square)
# Corrected print to list square names properly
print(f"Squares attacked by piece on D5 (Rook): {[chess.SQUARE_NAMES[sq] for sq in rook_attack_set]}")
print(f"Is D8 (King's square) in rook_attack_set: {king_square in rook_attack_set}")

if king_square not in rook_attack_set and board_attacks.is_attacked_by(chess.WHITE, king_square): # added a check that king is actually attacked
    print("ISSUE LIKELY CONFIRMED (board.attacks): King's square D8 is NOT in the attack set of the Rook on D5, though the king IS attacked by white.")
elif king_square in rook_attack_set:
    print("King's square D8 IS in the attack set of the Rook on D5. No issue observed here with board.attacks().")
else:
    print("King's square D8 is NOT in the attack set, AND board.is_attacked_by() also indicates no attack by white on D8. Check FEN or logic if this is unexpected.")
print("-" * 30 + "\n")