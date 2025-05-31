import os
stockfish_path = "/usr/games/stockfish"
exists = os.path.exists(stockfish_path)
print(f"Does '{stockfish_path}' exist? {exists}")
if exists:
    print(f"Is it executable? {os.access(stockfish_path, os.X_OK)}")