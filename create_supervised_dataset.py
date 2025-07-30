# FILENAME: create_supervised_dataset.py

import chess
import chess.pgn
import json
from tqdm import tqdm
from pathlib import Path
import logging
import argparse

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_outcome_from_result(result_str):
    """
    Converts a PGN result string to a numerical outcome from White's perspective.
    - White win ('1-0'): 1.0
    - Black win ('0-1'): -1.0
    - Draw ('1/2-1/2' or '*'): 0.0
    """
    if result_str == '1-0':
        return 1.0
    elif result_str == '0-1':
        return -1.0
    else: # "1/2-1/2" or "*"
        return 0.0

def create_supervised_dataset(pgn_path: Path, output_path: Path):
    """
    Parses a PGN corpus and creates a supervised learning dataset.

    The dataset is a .jsonl file where each line is a JSON object containing:
    - 'fen': The board state in FEN notation.
    - 'played_move': The move played from that state in UCI format.
    - 'outcome': The final game outcome from White's perspective (1.0, -1.0, 0.0).
    """
    try:
        if not pgn_path.exists():
            logging.error(f"FATAL: PGN corpus not found at '{pgn_path}'. Please ensure the file exists.")
            return

        logging.info(f"Starting dataset creation from '{pgn_path}'.")
        logging.info(f"Output will be saved to '{output_path}'.")

        position_count = 0
        game_count = 0

        # Use a robust encoding, latin-1 is good for many chess PGNs
        encoding = 'latin-1'

        with open(pgn_path, encoding=encoding) as pgn_file, open(output_path, 'w') as out_file:
            # First pass to count games for tqdm
            with open(pgn_path, encoding=encoding) as f_count:
                total_games = sum(1 for line in f_count if line.strip().startswith('[Event "'))
            pgn_file.seek(0) # Reset file pointer

            with tqdm(total=total_games, desc="Processing Games") as pbar:
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                    except (ValueError, UnicodeDecodeError) as e:
                        logging.warning(f"Skipping a malformed game due to parsing error: {e}")
                        continue

                    if game is None:
                        break
                    
                    game_count += 1
                    pbar.update(1)
                    
                    board = game.board()
                    result_str = game.headers.get("Result", "*")
                    # Get a single outcome for the whole game from White's perspective
                    outcome = get_outcome_from_result(result_str)

                    for move in game.mainline_moves():
                        fen = board.fen()
                        played_move_uci = move.uci()
                        
                        data_point = {
                            "fen": fen,
                            "played_move": played_move_uci,
                            "outcome": outcome # This is now consistent for the whole game
                        }
                        
                        out_file.write(json.dumps(data_point) + '\n')
                        position_count += 1
                        
                        board.push(move)

        logging.info("Dataset creation complete.")
        logging.info(f"Processed {game_count} games.")
        logging.info(f"Generated {position_count} total training data points.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a supervised learning dataset from a PGN file.")
    parser.add_argument("--input_pgn", required=True, help="Path to the input PGN file.")
    parser.add_argument("--output_jsonl", required=True, help="Path for the output JSON Lines (.jsonl) file.")
    args = parser.parse_args()

    input_path = Path(args.input_pgn)
    output_path = Path(args.output_jsonl)
    
    create_supervised_dataset(input_path, output_path)