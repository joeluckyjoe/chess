import chess
print(f"python-chess version: {chess.__version__}")

fen = "k7/8/8/3q4/8/8/3P4/3K4 w - - 0 1"
board = chess.Board(fen)

print(f"\nTesting FEN: {fen}")
print("Board state:")
print(board)

pd2_square = chess.D2
piece = board.piece_at(pd2_square)

if piece:
    print(f"\nPiece on {chess.square_name(pd2_square)}: {piece} (Color: {'WHITE' if piece.color == chess.WHITE else 'BLACK'})")
    is_pinned_by_library = board.is_pinned(chess.WHITE, pd2_square)
    print(f"Is board.is_pinned(chess.WHITE, {chess.square_name(pd2_square)}) True? : {is_pinned_by_library}")
else:
    print(f"\nNo piece found on {chess.square_name(pd2_square)} for FEN {fen}")

mobility_count = 0
print("\nLegal moves for the whole board (from board.legal_moves):")
if not list(board.legal_moves):
    print("  (No legal moves for current player)")
for move in board.legal_moves:
    print(f"  {move.uci()}")
    if move.from_square == pd2_square:
        mobility_count += 1

print(f"\nCalculated mobility for piece on {chess.square_name(pd2_square)}: {mobility_count}")