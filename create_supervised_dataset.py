# FILENAME: create_supervised_dataset.py

import chess
import chess.pgn
import json
from tqdm import tqdm
from pathlib import Path
import logging

# Assuming config.py is in the parent directory or accessible in the Python path
from config import get_paths

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_outcome_from_result(result_str, player_turn):
    """
    Converts a PGN result string to a numerical outcome from the perspective of the current player.
    - Win: 1.0
    - Loss: -1.0
    - Draw: 0.0
    
    Args:
        result_str (str): The 'Result' header from the PGN game (e.g., '1-0', '0-1', '1/2-1/2').
        player_turn (chess.Color): The color of the player whose turn it is (chess.WHITE or chess.BLACK).

    Returns:
        float: The numerical outcome.
    """
    if result_str == '1-0':
        return 1.0 if player_turn == chess.WHITE else -1.0
    elif result_str == '0-1':
        return -1.0 if player_turn == chess.WHITE else 1.0
    elif result_str == '1/2-1/2':
        return 0.0
    else:
        return 0.0 # Treat '*' or other results as draws

def create_kasparov_dataset():
    """
    Parses the Kasparov PGN corpus and creates a supervised learning dataset.

    The dataset is a .jsonl file where each line is a JSON object containing:
    - 'fen': The board state in FEN notation.
    - 'played_move': The move played from that state in UCI format.
    - 'outcome': The final game outcome from the current player's perspective (1.0, -1.0, 0.0).
    """
    try:
        paths = get_paths()
        pgn_corpus_path = paths.drive_project_root / 'Kasparov.pgn'
        output_dataset_path = paths.drive_project_root / 'kasparov_supervised_dataset.jsonl'

        if not pgn_corpus_path.exists():
            logging.error(f"FATAL: PGN corpus not found at '{pgn_corpus_path}'. Please ensure the file exists.")
            return

        logging.info(f"Starting dataset creation from '{pgn_corpus_path}'.")
        logging.info(f"Output will be saved to '{output_dataset_path}'.")

        position_count = 0
        game_count = 0

        with open(pgn_corpus_path, encoding='latin-1') as pgn_file, open(output_dataset_path, 'w') as out_file:
            # First pass to count games for tqdm
            with open(pgn_corpus_path, encoding='latin-1') as f_count:
                total_games = sum(1 for line in f_count if line.startswith('[Event "'))
            pgn_file.seek(0) # Reset file pointer

            with tqdm(total=total_games, desc="Processing Games") as pbar:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    game_count += 1
                    pbar.update(1)
                    
                    board = game.board()
                    result_str = game.headers.get("Result", "*")

                    for move in game.mainline_moves():
                        fen = board.fen()
                        played_move_uci = move.uci()
                        
                        # Get outcome from the perspective of the player about to move
                        outcome = get_outcome_from_result(result_str, board.turn)

                        data_point = {
                            "fen": fen,
                            "played_move": played_move_uci,
                            "outcome": outcome
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
    create_kasparov_dataset()