#
# File: evaluate_on_puzzles.py (Corrected for Phase AO)
#
import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import chess
from torch_geometric.data import Batch

# --- Setup Python Path ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Project-specific Imports ---
from config import get_paths
from gnn_agent.neural_network.gnn_models import ChessNetwork
from gnn_agent.gamestate_converters import gnn_data_converter, action_space_converter

def load_model_from_checkpoint(model_path: str, device: torch.device) -> ChessNetwork:
    """
    Loads a ChessNetwork model from a .pth checkpoint file, using the
    correct, up-to-date network architecture.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    # --- FIX: Instantiate the network with the correct, current architecture ---
    # This matches the architecture used in the training script.
    network = ChessNetwork(
        embed_dim=256,
        gnn_hidden_dim=128,
        num_heads=4
    ).to(device)
    # --- END FIX ---

    checkpoint = torch.load(model_path, map_location=device)
    
    # This handles checkpoints saved by different scripts
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    network.load_state_dict(state_dict)
    network.eval() # Set the model to evaluation mode
    
    print(f"✅ Successfully loaded model for evaluation from {os.path.basename(str(model_path))}")
    return network


def get_agent_top_move(board: chess.Board, network: ChessNetwork, device: torch.device) -> str:
    """
    Gets the agent's top-ranked move for a given board position.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    # --- FIX: Use the torch_geometric Batch object to correctly format a single item ---
    # This ensures the data has the correct shape and attributes (like .batch and .piece_batch)
    # that the network's forward pass expects.
    single_data = gnn_data_converter.convert_to_gnn_input(board, device='cpu')
    
    # Manually create the piece_batch attribute for a single graph
    num_pieces = single_data.piece_features.size(0)
    single_data.piece_batch = torch.zeros(num_pieces, dtype=torch.long)
    
    # Use Batch.from_data_list to create a batch of size 1
    batch = Batch.from_data_list([single_data]).to(device)
    # --- END FIX ---

    with torch.no_grad():
        # --- FIX: Call the network with the single batch object ---
        policy_logits, _ = network(batch)
        # --- END FIX ---

    policy_probs = torch.softmax(policy_logits.flatten(), dim=0)
    
    valid_legal_moves = []
    legal_move_indices_list = []
    
    for m in legal_moves:
        try:
            idx = action_space_converter.move_to_index(m, board)
            valid_legal_moves.append(m)
            legal_move_indices_list.append(idx)
        except ValueError:
            # This can happen if a move is somehow not in our defined action space
            # which is rare but possible.
            print(f"Warning: Skipping move with invalid index. Move: {m.uci()}")

    if not valid_legal_moves:
        print("ERROR: No legal moves could be mapped to a valid index.")
        return ""

    legal_move_indices = torch.tensor(legal_move_indices_list, device=device, dtype=torch.long)
    legal_probs = torch.gather(policy_probs, 0, legal_move_indices)
    
    best_move_index_in_legal_list = torch.argmax(legal_probs)
    best_move = valid_legal_moves[best_move_index_in_legal_list]

    return best_move.uci()


def run_evaluation(model_path: str, puzzle_file_path: str, device: torch.device):
    """ Main loop to run evaluation against a puzzle file. """
    network = load_model_from_checkpoint(model_path, device)

    if not os.path.exists(puzzle_file_path):
        raise FileNotFoundError(f"Puzzle file not found: {puzzle_file_path}")

    with open(puzzle_file_path, 'r') as f:
        puzzles = [json.loads(line) for line in f]

    if not puzzles:
        print("No puzzles found in the file. Exiting.")
        return

    puzzles_solved = 0
    total_puzzles = len(puzzles)
    print(f"\nStarting evaluation on {total_puzzles} puzzles...")

    for i, puzzle in enumerate(puzzles):
        fen = puzzle['fen']
        solution_move_uci = puzzle['best_move']
        board = chess.Board(fen)
        
        agent_move_uci = get_agent_top_move(board, network, device)
        is_correct = (agent_move_uci == solution_move_uci)
        
        if is_correct:
            puzzles_solved += 1
        
        result_str = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        if not agent_move_uci:
            result_str = "❌ ERROR (No valid move produced)"

        print(f"Puzzle {i+1}/{total_puzzles}: Agent chose {agent_move_uci}, solution is {solution_move_uci}.   [{result_str}]")

    success_rate = (puzzles_solved / total_puzzles) * 100 if total_puzzles > 0 else 0
    
    print("\n-------------------------------------------")
    print("         Puzzle Evaluation Complete")
    print("-------------------------------------------")
    print(f"Puzzles Solved: {puzzles_solved} / {total_puzzles}")
    print(f"Success Rate:   {success_rate:.2f}%")
    print("-------------------------------------------")


def main():
    """Parses arguments and launches the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a chess agent on tactical puzzles.")
    parser.add_argument("--model", type=str, required=True, help="Path to a specific model checkpoint.")
    parser.add_argument("--puzzles", type=str, required=True, help="Path to the puzzle file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use Path for robust path handling
    model_path = Path(args.model)
    puzzle_path = Path(args.puzzles)

    run_evaluation(str(model_path), str(puzzle_path), device)


if __name__ == '__main__':
    main()
