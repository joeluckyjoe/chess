import chess
import chess.pgn
import json
from tqdm import tqdm
from pathlib import Path
import logging
import argparse

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_style_dataset(pgn_path: Path, output_path: Path, player_name: str, skip_moves: int):
    """
    Parses a PGN corpus and creates a style-based dataset, skipping early moves.
    """
    try:
        if not pgn_path.exists():
            logging.error(f"FATAL: PGN corpus not found at '{pgn_path}'.")
            return

        logging.info(f"Starting style dataset creation for player: '{player_name}'")
        logging.info(f"Skipping the first {skip_moves} half-moves of each game.")
        logging.info(f"Reading from '{pgn_path}' and writing to '{output_path}'.")

        position_count = 0
        game_count = 0
        win_count = 0
        encoding = 'latin-1'

        with open(pgn_path, encoding=encoding) as pgn_file, open(output_path, 'w') as out_file:
            with open(pgn_path, encoding=encoding) as f_count:
                total_games = sum(1 for line in f_count if line.strip().startswith('[Event "'))
            pgn_file.seek(0)

            with tqdm(total=total_games, desc="Processing Games") as pbar:
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                    except Exception as e:
                        logging.warning(f"Skipping a malformed game due to parsing error: {e}")
                        continue

                    if game is None:
                        break
                    
                    game_count += 1
                    pbar.update(1)
                    
                    headers = game.headers
                    result = headers.get("Result", "*")
                    white_player = headers.get("White", "").strip()
                    black_player = headers.get("Black", "").strip()

                    player_is_white = player_name in white_player
                    player_is_black = player_name in black_player
                    
                    player_won = (player_is_white and result == "1-0") or \
                                 (player_is_black and result == "0-1")

                    if not player_won:
                        continue
                    
                    win_count += 1
                    board = game.board()
                    
                    # --- MODIFIED: Loop through moves with an index ---
                    for move_index, move in enumerate(game.mainline_moves()):
                        # --- MODIFIED: Skip the first N moves (openings) ---
                        if move_index < skip_moves:
                            board.push(move)
                            continue

                        is_players_turn = (board.turn == chess.WHITE and player_is_white) or \
                                          (board.turn == chess.BLACK and player_is_black)
                        
                        if is_players_turn:
                            data_point = {
                                "fen": board.fen(),
                                "label": f"{player_name.lower().split(',')[0]}_win"
                            }
                            out_file.write(json.dumps(data_point) + '\n')
                            position_count += 1
                        
                        board.push(move)

        logging.info("Dataset creation complete.")
        logging.info(f"Processed {game_count} total games.")
        logging.info(f"Found and processed {win_count} wins for '{player_name}'.")
        logging.info(f"Generated {position_count} total training data points from middlegames.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a style-based dataset from a PGN file.")
    parser.add_argument("--input-pgn", required=True, help="Path to the input PGN file.")
    parser.add_argument("--output-jsonl", required=True, help="Path for the output JSON Lines file.")
    parser.add_argument("--player-name", required=True, help="The name of the player to filter for.")
    # --- MODIFIED: Added argument to skip opening moves ---
    parser.add_argument("--skip-moves", type=int, default=20, help="Number of half-moves to skip from the start of each game (default: 20, i.e., 10 full moves).")
    args = parser.parse_args()

    input_path = Path(args.input_pgn)
    output_path = Path(args.output_jsonl)
    
    create_style_dataset(input_path, output_path, args.player_name, args.skip_moves)