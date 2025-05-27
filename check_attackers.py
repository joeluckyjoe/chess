import chess
print(f"python-chess version: {chess.__version__}")

fen = "3k4/3p4/5n2/3R2B1/4P3/8/8/K7 w - - 0 1"
board = chess.Board(fen)

current_piece_square = chess.D5         # The Rook's square
current_piece_color = chess.WHITE       # The Rook's color
opponent_king_color = chess.BLACK

print(f"\nTesting FEN: {fen}")
print(f"Board state:\n{board}")

opponent_king_square = board.king(opponent_king_color)

if opponent_king_square is None:
    print(f"\nError: Opponent King (Color: {opponent_king_color}) not found on board!")
else:
    print(f"\nOpponent King (Color: {opponent_king_color}) is on square: {chess.square_name(opponent_king_square)}")
    
    # Get attackers of the opponent king's square by pieces of current_piece_color
    attackers_of_king = board.attackers(current_piece_color, opponent_king_square)
    print(f"Squares of '{chess.COLOR_NAMES[current_piece_color]}' pieces attacking '{chess.square_name(opponent_king_square)}': {[chess.square_name(s) for s in attackers_of_king]}")
    
    is_current_piece_an_attacker = current_piece_square in attackers_of_king
    print(f"Is the piece on '{chess.square_name(current_piece_square)}' in this set of attackers? : {is_current_piece_an_attacker}")

    if is_current_piece_an_attacker:
        print("Conclusion: The piece on d5 IS attacking the king on d8. 'is_checking' should be 1.")
    else:
        print("Conclusion: The piece on d5 IS NOT attacking the king on d8 (according to board.attackers). 'is_checking' would be 0.")