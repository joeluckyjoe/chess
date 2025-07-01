#
# File: analyze_corpus.py
#
"""
Processes a master 'analysis_corpus.jsonl' file to generate a quantitative
summary of the agent's performance and weaknesses compared to Stockfish.

This script calculates several key metrics:
- Value Misalignment: How different the agent's evaluation is from Stockfish's.
- Top Move Agreement: How often the agent's best move matches Stockfish's.
- Blunder Rate: The percentage of moves that cause a significant drop in win probability.

These metrics are broken down by game phase (Opening, Middlegame, Endgame) to
provide a detailed diagnostic report.
"""
import argparse
import json
import numpy as np
import sys
import os
from pathlib import Path
from collections import defaultdict

import chess
import chess.engine

# Add project root to path to allow importing from config
project_root_path = Path(os.path.abspath(__file__)).parent
if str(project_root_path) not in sys.path:
    # Handle cases where the script might be in a subdirectory like 'analysis_scripts'
    if project_root_path.name == "analysis_scripts":
        project_root_path = project_root_path.parent
    sys.path.insert(0, str(project_root_path))

from config import config_params

# --- Constants for Analysis ---
BLUNDER_THRESHOLD_CP = 150  # A 1.5 pawn swing is a blunder
MATE_SCORE = 10000          # Arbitrary large number for mates

def parse_stockfish_eval(eval_str: str, pov_color: chess.Color) -> float:
    """
    Parses the evaluation string from the log into a consistent centipawn score.
    The score is always from the perspective of the current player (pov_color).
    """
    if not isinstance(eval_str, str) or "N/A" in eval_str or "Error" in eval_str:
        return 0.0

    if "Mate" in eval_str:
        try:
            mate_in = int(eval_str.split('(')[1].replace(')', ''))
            # Positive mate is good, negative mate is bad.
            # A shorter mate is better/worse than a longer one.
            score = MATE_SCORE - abs(mate_in)
            return score if mate_in > 0 else -score
        except (ValueError, IndexError):
             return 0.0 # Handle malformed "Mate" strings

    try:
        # The log saves from White's perspective. Adjust for the current player.
        score = float(eval_str)
        return score if pov_color == chess.WHITE else -score
    except (ValueError, TypeError):
        return 0.0

def get_game_phase(board: chess.Board) -> str:
    """Determines the game phase based on the number of pieces."""
    # Heuristic: Count pieces, excluding kings
    num_pieces = chess.popcount(board.occupied & ~board.kings)
    if num_pieces > 20:
        return "opening"
    elif num_pieces <= 8:
        return "endgame"
    else:
        return "middlegame"

def analyze_corpus(corpus_path: Path, stockfish_path: str):
    """
    Main analysis function.
    """
    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        print("Stockfish engine started successfully.")
    except Exception as e:
        print(f"Error: Could not start Stockfish engine at '{stockfish_path}'. {e}")
        return

    # Data structures to hold our metrics, categorized by game phase
    phases = ["opening", "middlegame", "endgame", "total"]
    metrics = {phase: defaultdict(list) for phase in phases}
    
    total_moves = 0

    print(f"\nProcessing corpus file: {corpus_path.name}...")
    with open(corpus_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line: {line.strip()}")
                continue

            # Skip metadata line or entries without agent eval
            if entry.get("type") == "analysis_metadata" or "agent_eval" not in entry:
                continue

            total_moves += 1
            board = chess.Board(entry["board_fen_before"])
            phase = get_game_phase(board)

            # --- 1. Value Misalignment ---
            agent_value_tanh = entry["agent_eval"]["value_before"] # Range [-1, 1]
            stockfish_cp_before = parse_stockfish_eval(entry["stockfish_eval"], board.turn)
            # Simple conversion: map tanh value to a centipawn range for comparison
            agent_value_cp = agent_value_tanh * MATE_SCORE
            value_error = abs(agent_value_cp - stockfish_cp_before)
            metrics[phase]["value_error"].append(value_error)

            # --- 2. Top Move Agreement & 3. Blunder Rate ---
            agent_move_uci = entry["move_uci"]
            agent_move = chess.Move.from_uci(agent_move_uci)

            # Get Stockfish's opinion on the position
            info = engine.analyse(board, chess.engine.Limit(depth=12))
            stockfish_best_move = info["pv"][0] if "pv" in info and info["pv"] else None

            # Agreement
            if stockfish_best_move and agent_move == stockfish_best_move:
                metrics[phase]["top_move_agreements"].append(1)
            else:
                metrics[phase]["top_move_agreements"].append(0)

            # Blunder check
            # We need the evaluation *after* the agent's move
            board.push(agent_move)
            stockfish_cp_after_info = engine.analyse(board, chess.engine.Limit(depth=10))
            
            # --- BUG FIX ---
            # Directly process the live score object instead of parsing a string
            score_obj_after = stockfish_cp_after_info.get("score")
            stockfish_cp_after = 0.0
            if score_obj_after:
                # The POV is for the player whose turn it is now (i.e., opponent)
                pov_score_after = score_obj_after.pov(board.turn)
                if pov_score_after.is_mate():
                    # Positive mate for the opponent is a very bad score for us.
                    stockfish_cp_after = -(MATE_SCORE - abs(pov_score_after.mate()))
                else:
                    # score() is from white's POV. pov().score() is from the current player's.
                    # A positive score for the opponent is a negative score for us.
                    stockfish_cp_after = -(pov_score_after.score(mate_score=MATE_SCORE) or 0)
            # --- END FIX ---
            
            # Change in score is from the perspective of the player *who just moved*
            pov_change = stockfish_cp_after - stockfish_cp_before
            
            if pov_change < -BLUNDER_THRESHOLD_CP:
                 metrics[phase]["blunders"].append(1)
            else:
                 metrics[phase]["blunders"].append(0)

    if engine:
        engine.quit()

    # --- Aggregate and Print Report ---
    print("\n--- Agent Performance Analysis Report ---")
    print(f"Total moves analyzed: {total_moves}\n")
    
    # Calculate totals first
    for phase in ["opening", "middlegame", "endgame"]:
        for key, values in metrics[phase].items():
            metrics["total"][key].extend(values)

    for phase in phases:
        num_positions = len(metrics[phase]["value_error"])
        if num_positions == 0:
            continue
            
        avg_value_error = np.mean(metrics[phase]["value_error"])
        agreement_rate = np.mean(metrics[phase]["top_move_agreements"]) * 100
        blunder_rate = np.mean(metrics[phase]["blunders"]) * 100

        print(f"--- {phase.upper()} ({num_positions} positions) ---")
        print(f"  - Avg. Value Misalignment: {avg_value_error:,.0f} CP")
        print(f"  - Top Move Agreement Rate: {agreement_rate:.1f}%")
        print(f"  - Blunder Rate:            {blunder_rate:.1f}%\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a JSONL corpus of game data to quantify agent performance.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "corpus_path",
        type=str,
        help="Path to the analysis_corpus.jsonl file."
    )
    
    stockfish_default_path = config_params.get("STOCKFISH_PATH")
    parser.add_argument(
        "--stockfish_path",
        type=str,
        default=stockfish_default_path,
        help="Path to the Stockfish executable.\nDefaults to the path in your config.py."
    )
    
    args = parser.parse_args()

    if not args.stockfish_path or not os.path.exists(args.stockfish_path):
        print("---")
        print("ERROR: Stockfish path not found or not provided.")
        print("Please provide a valid path using the --stockfish_path argument")
        print(f"or ensure it is set correctly in your config.py file.")
        print("---")
        sys.exit(1)

    analyze_corpus(Path(args.corpus_path), args.stockfish_path)


if __name__ == "__main__":
    main()