# FILENAME: generate_puzzles_from_corpus.py
"""
Scans a directory of PGN files to find and export tactical puzzles based on blunders.

This script iterates through the N most recent PGN files in a specified directory,
analyzes each game move-by-move using Stockfish, and identifies blunders.
A blunder is defined as a move that causes a significant drop in evaluation
compared to the best possible move.

Each identified puzzle (the board state before the blunder and the correct move)
is saved to a single JSONL file, which can then be used to train the agent.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import chess
import chess.pgn
import chess.engine

# Add the project root to the Python path to allow for module imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config_params, get_paths

def generate_puzzles(args):
    """
    Main logic for scanning PGNs and generating blunder-based puzzles.
    """
    paths = get_paths()
    stockfish_path = config_params.get("STOCKFISH_PATH")
    if not stockfish_path or not os.path.exists(stockfish_path):
        print(f"Error: Stockfish path not set or found. Path was: {stockfish_path}")
        return

    input_dir = Path(args.pgn_dir)
    if not input_dir.is_dir():
        print(f"Error: PGN directory not found at {input_dir}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all PGN files and sort them by modification time (most recent first)
    try:
        all_pgns = [p for p in input_dir.glob('*.pgn') if p.is_file()]
        all_pgns.sort(key=os.path.getmtime, reverse=True)
    except Exception as e:
        print(f"Error reading PGN directory: {e}")
        return

    files_to_process = all_pgns[:args.num_games]

    if not files_to_process:
        print(f"No PGN files found in {input_dir}. Exiting.")
        return

    print(f"Found {len(all_pgns)} PGN files. Processing the {len(files_to_process)} most recent ones.")
    print(f"Analysis Depth: {args.depth} | Blunder Threshold: {args.blunder_threshold}cp")
    print(f"Output will be appended to: {output_path}")
    print("-" * 60)

    stockfish_engine = None
    puzzles_found_total = 0
    
    # A large centipawn value to represent a checkmate advantage
    MATE_SCORE = 100000 

    try:
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        for pgn_file_path in files_to_process:
            print(f"Scanning file: {pgn_file_path.name}...")
            puzzles_in_file = 0
            
            with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        # Iterate through nodes to have access to board state before the move
                        for node in game.mainline():
                            if node.parent is None: # Skip the root node
                                continue

                            board_before_move = node.parent.board()
                            move_played = node.move
                            
                            # 1. Analyze the position BEFORE the move
                            info_before = stockfish_engine.analyse(board_before_move, chess.engine.Limit(depth=args.depth))
                            
                            if 'score' not in info_before or 'pv' not in info_before or not info_before['pv']:
                                continue
                            
                            best_move_found = info_before['pv'][0]
                            
                            # Skip if the move played was actually the best move
                            if move_played == best_move_found:
                                continue

                            eval_best_pov = info_before['score'].pov(board_before_move.turn)
                            
                            # 2. Analyze the position AFTER the move
                            board_after_move = node.board()
                            info_after = stockfish_engine.analyse(board_after_move, chess.engine.Limit(depth=args.depth - 2))

                            if 'score' not in info_after:
                                continue

                            eval_played_pov = info_after['score'].pov(board_after_move.turn).opponent()
                            
                            # 3. Compare evaluations and check for a blunder
                            eval_best_cp = eval_best_pov.score(mate_score=MATE_SCORE)
                            eval_played_cp = eval_played_pov.score(mate_score=MATE_SCORE)
                            eval_drop = eval_best_cp - eval_played_cp
                            
                            if eval_drop >= args.blunder_threshold:
                                puzzles_in_file += 1
                                puzzle = {"fen": board_before_move.fen(), "best_move": best_move_found.uci()}
                                with open(output_path, 'a') as f:
                                    f.write(json.dumps(puzzle) + '\n')

                    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, ValueError) as e:
                        print(f"\nStockfish analysis failed: {e}. Restarting engine.")
                        stockfish_engine.quit()
                        stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                    except Exception as e:
                        print(f"\nAn unexpected error occurred while processing a game: {e}")
                        continue
            
            if puzzles_in_file > 0:
                print(f"  -> Found {puzzles_in_file} new puzzles in this file.")
                puzzles_found_total += puzzles_in_file

    finally:
        if stockfish_engine:
            stockfish_engine.quit()
            print("\nStockfish engine terminated.")
    
    print("-" * 60)
    print("Corpus analysis complete.")
    print(f"Found a total of {puzzles_found_total} new puzzles.")
    print(f"All puzzles have been appended to {output_path}")

def main():
    """
    Parses command-line arguments and runs the puzzle generation process.
    """
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Scan a corpus of PGNs to generate blunder-based puzzles.")
    
    parser.add_argument("--pgn-dir", type=str, default=str(paths.pgn_games_dir),
                        help=f"Directory containing PGN files to analyze. Default: {paths.pgn_games_dir}")
    
    parser.add_argument("--output", type=str, default=str(paths.generated_puzzles_file),
                        help=f"Path to the output JSONL file for appending puzzles. Default: {paths.generated_puzzles_file}")
    
    parser.add_argument("-n", "--num-games", type=int, default=20,
                        help="Number of the most recent games to process. Default: 20")
                        
    parser.add_argument("--blunder-threshold", type=int, default=200,
                        help="The minimum evaluation drop in centipawns to be considered a blunder. Default: 200.")
                        
    parser.add_argument("--depth", type=int, default=20,
                        help="The depth for the primary Stockfish analysis. Default: 20.")

    args = parser.parse_args()
    generate_puzzles(args)

if __name__ == '__main__':
    main()