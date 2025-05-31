# temp_fen_test.py
import chess

# Create a new board in the starting position
board = chess.Board()
print(f"Initial FEN: {board.fen()}")

# Apply the move e2e4
move = chess.Move.from_uci("e2e4")
board.push(move)

# Print the FEN after the move
print(f"FEN after e2e4: {board.fen()}")

# Expected FEN after e2e4: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
# Let's specifically check the en passant square from the board object
if board.ep_square:
    print(f"En passant square available at: {chess.square_name(board.ep_square)}")
else:
    print("No en passant square available.")

# Apply the move c7c5
move2 = chess.Move.from_uci("c7c5")
board.push(move2)
print(f"\nFEN after e2e4, c7c5: {board.fen()}")
if board.ep_square:
    print(f"En passant square after c7c5: {chess.square_name(board.ep_square)}") # Should be c6
else:
    print("No en passant square after c7c5.")
