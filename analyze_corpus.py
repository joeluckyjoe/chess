#
# File: analyze_corpus.py
#
"""
Analyzes a structured JSONL corpus of game data to identify patterns
of agent weakness.

Usage:
python analyze_corpus.py [--corpus_path /path/to/your/corpus.jsonl]
"""
import json
import argparse
import re
from pathlib import Path

# Import the centralized path management function
from config import get_paths

def parse_stockfish_mate_eval(eval_str: str):
    """
    Parses a Stockfish evaluation string to find a mate-in-N value.
    Example: "Mate(3)" -> 3, "Mate(-2)" -> -2.
    Returns None if no mate is found.
    """
    if not isinstance(eval_str, str):
        return None
    
    match = re.match(r"Mate\((-?\d+)\)", eval_str)
    if match:
        return int(match.group(1))
    return None

def find_actual_mating_move(board):
    """
    Checks ALL legal moves on the board to find one that is an immediate checkmate.
    This is only effective for finding a mate-in-1.
    Returns the SAN of the mating move if found, otherwise None.
    """
    # Create a copy to not alter the original board passed to the function
    board_copy = board.copy()
    for move in board_copy.legal_moves:
        # Create a second copy to test each move
        temp_board = board_copy.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            # Return the Standard Algebraic Notation (SAN) of the move
            return board_copy.san(move)
    return None

def analyze_checkmate_blindness(corpus_path: Path):
    """
    Scans the corpus for positions where a forced mate exists but the agent
    does not select the move that continues the mating sequence.
    """
    print(f"--- Analyzing Corpus for Checkmate Blindness: {corpus_path.name} ---")

    # This import is here because it's only needed for this analysis function
    try:
        import chess
    except ImportError:
        print("Error: The 'python-chess' library is required. Please install it with 'pip install chess'")
        return

    blindness_events = []
    current_pgn_file = "N/A"

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                log_entry = json.loads(line)

                # Capture metadata. It applies to all subsequent log entries
                # until a new metadata line is found.
                if log_entry.get("type") == "analysis_metadata":
                    current_pgn_file = log_entry.get("pgn_file", "N/A")
                    continue # Skip to the next line

                # --- Core Analysis Logic ---
                stockfish_eval = log_entry.get("stockfish_eval")
                mate_in_n = parse_stockfish_mate_eval(stockfish_eval)

                # A mate is available if mate_in_n is not None
                if mate_in_n is not None:
                    board = chess.Board(log_entry.get("board_fen_before"))
                    
                    # Check if it's our agent's turn to deliver the mate
                    is_agent_mate = (board.turn == chess.WHITE and mate_in_n > 0) or \
                                      (board.turn == chess.BLACK and mate_in_n < 0)

                    if is_agent_mate:
                        agent_policy = log_entry.get("agent_eval", {}).get("policy_before", [])
                        if not agent_policy:
                            continue

                        agent_top_move = agent_policy[0]["move"]
                        
                        # We determine blindness if the agent's move is not the mating move.
                        board_after_top_move = board.copy()
                        try:
                            board_after_top_move.push_san(agent_top_move)
                        except (ValueError, chess.IllegalMoveError):
                            pass # An illegal move is clearly not the mating move.

                        if not board_after_top_move.is_checkmate():
                            # BLINDNESS DETECTED!
                            
                            actual_mating_move = "Unknown"
                            if abs(mate_in_n) == 1:
                                # For M1, we can find the exact move.
                                actual_mating_move = find_actual_mating_move(board) or "Error: Mate-in-1 move not found"
                            else:
                                # For M > 1, we can't know the exact move from the log alone.
                                actual_mating_move = f"(Should start Mate-in-{abs(mate_in_n)} sequence)"

                            event = {
                                "ply": log_entry.get("ply"),
                                "fen": log_entry.get("board_fen_before"),
                                "stockfish_eval": stockfish_eval,
                                "agent_top_move": agent_top_move,
                                "agent_top_prob": agent_policy[0]["prob"],
                                "actual_mating_move": actual_mating_move,
                                "pgn_source": current_pgn_file
                            }
                            blindness_events.append(event)
            
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping malformed or incomplete log entry on line {line_num}. Error: {e}")

    # --- Print Report ---
    if not blindness_events:
        print("\n✅ Analysis Complete. No checkmate blindness events found.")
    else:
        print(f"\n🚨 Analysis Complete. Found {len(blindness_events)} checkmate blindness events:")
        for i, event in enumerate(blindness_events, 1):
            print(f"\n--- Event {i} (from PGN: {event['pgn_source']}) ---")
            print(f"  Position (FEN): {event['fen']}")
            print(f"  Ply: {event['ply']}")
            print(f"  Stockfish found: {event['stockfish_eval']}")
            print(f"  Agent's Top Move: '{event['agent_top_move']}' (Prob: {event['agent_top_prob']:.3f})")
            print(f"  Correct Action: '{event['actual_mating_move']}'")
            print("-" * (20 + len(str(event['pgn_source']))))


def main():
    """
    Main function to parse arguments and run the analysis.
    """
    # Get the correct project paths from the central config
    paths = get_paths()
    # The default corpus path should point to the output of create_corpus.py
    # Access the path by attribute name, not dictionary key.
    default_corpus_path = paths.analysis_output_dir / 'analysis_corpus.jsonl'
    
    parser = argparse.ArgumentParser(
        description="Analyze a JSONL corpus for agent weaknesses.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=default_corpus_path,
        help=f"Path to the JSONL corpus file to analyze.\nDefaults to: {default_corpus_path}"
    )
    args = parser.parse_args()

    if not args.corpus_path.exists():
        print(f"Error: Corpus file not found at '{args.corpus_path}'")
        print("Please run 'create_corpus.py' first to generate the analysis corpus.")
        return

    analyze_checkmate_blindness(args.corpus_path)


if __name__ == '__main__':
    main()
