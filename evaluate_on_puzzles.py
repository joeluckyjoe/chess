#
# File: evaluate_on_puzzles.py (Modified for Diagnostics)
#
"""
A script to evaluate a trained agent's performance on a specific set of tactical puzzles.

This script loads the latest model checkpoint, reads a .jsonl file of puzzles
(where each line contains a FEN and the UCI representation of the best move),
and calculates the percentage of puzzles the agent solves correctly.
"""
import sys
import os
import json
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import chess
from torch_geometric.data import Batch # <--- 1. IMPORT BATCH

# --- Project-specific Imports ---
from config import get_paths, config_params
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters import gnn_data_converter, action_space_converter


def find_latest_checkpoint(checkpoint_dir):
    """Finds the most recently modified checkpoint file in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth.tar'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def load_model_from_checkpoint(model_path, device):
    """Loads a ChessNetwork model from a .pth checkpoint file."""
    # This architecture must match the one used during training.
    square_gnn = SquareGNN(in_features=19, hidden_features=256, out_features=128, heads=4)
    piece_gnn = PieceGNN(in_channels=12, hidden_channels=256, out_channels=128)
    attention_module = CrossAttentionModule(sq_embed_dim=128, pc_embed_dim=128, num_heads=4, dropout_rate=0.1)
    policy_head = PolicyHead(embedding_dim=128, num_possible_moves=4672)
    value_head = ValueHead(embedding_dim=128)
    network = ChessNetwork(square_gnn, piece_gnn, attention_module, policy_head, value_head).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    network.load_state_dict(state_dict)
    network.eval()  # Set the model to evaluation mode
    print(f"✅ Successfully loaded model for evaluation from {os.path.basename(str(model_path))}")
    return network


def get_agent_top_move(board: chess.Board, network: ChessNetwork, device: torch.device) -> str:
    """
    Gets the agent's top-ranked move for a given board position.
    
    Returns:
        The agent's best move in UCI format (e.g., 'e2e4').
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    # --- 2. WRAP THE SINGLE DATA OBJECT INTO A BATCH ---
    single_data = gnn_data_converter.convert_to_gnn_input(board, device=device)
    batch_data = Batch.from_data_list([single_data])
    
    with torch.no_grad():
        # Pass the entire batch object to the network
        policy_logits, _ = network(batch_data)

    policy_probs = torch.softmax(policy_logits.flatten(), dim=0)
    
    # --- DIAGNOSTIC MODIFICATION ---
    # Safely get indices and filter out any moves that don't map to a valid index.
    valid_legal_moves = []
    legal_move_indices_list = []
    action_space_size = policy_probs.shape[0]

    for m in legal_moves:
        idx = action_space_converter.move_to_index(m, board)
        if 0 <= idx < action_space_size:
            valid_legal_moves.append(m)
            legal_move_indices_list.append(idx)
        else:
            # This print statement is crucial for debugging
            print(f"⚠️ DIAGNOSTIC: Ignoring move with invalid index. Move: {m.uci()}, Generated Index: {idx}")

    # If, after filtering, no legal moves can be mapped, we cannot proceed for this position.
    if not valid_legal_moves:
        print("❌ ERROR: No legal moves could be mapped to a valid index. Aborting this puzzle.")
        return "" # Return an empty string to signify failure

    legal_move_indices = torch.tensor(legal_move_indices_list, device=device, dtype=torch.long)

    # Gather the probabilities of only the VALID legal moves
    legal_probs = torch.gather(policy_probs, 0, legal_move_indices)

    # Find the index of the best move *within the list of VALID legal moves*
    best_move_index_in_legal_list = torch.argmax(legal_probs)
    
    # Get the corresponding move object from the now-validated list
    best_move = valid_legal_moves[best_move_index_in_legal_list]

    return best_move.uci()


def run_evaluation(model_path: str, puzzle_file_path: str, device: torch.device):
    """
    Main loop to run evaluation against a puzzle file.
    """
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
        # Handle the case where the agent couldn't produce a move
        if not agent_move_uci:
            result_str = "❌ ERROR (No valid move produced)"

        print(f"Puzzle {i+1}/{total_puzzles}: Agent chose {agent_move_uci}, solution is {solution_move_uci}.  [{result_str}]")

    # --- Final Report ---
    success_rate = (puzzles_solved / total_puzzles) * 100 if total_puzzles > 0 else 0
    
    print("\n-------------------------------------------")
    print("        Puzzle Evaluation Complete")
    print("-------------------------------------------")
    print(f"Puzzles Solved: {puzzles_solved} / {total_puzzles}")
    print(f"Success Rate:   {success_rate:.2f}%")
    print("-------------------------------------------")


def main():
    """Parses arguments and launches the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a chess agent on tactical puzzles.")
    parser.add_argument("--model", type=str, default=None, help="Path to a specific model checkpoint. If not provided, the latest will be used.")
    parser.add_argument("--puzzles", type=str, default="puzzles_eval.jsonl", help="Name of the puzzle file relative to the project root.")
    args = parser.parse_args()

    # To get a clearer error message from CUDA, we run it in blocking mode.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = get_paths()
    
    model_to_evaluate = args.model
    if not model_to_evaluate:
        print("No specific model provided, finding the latest checkpoint...")
        model_to_evaluate = find_latest_checkpoint(paths.checkpoints_dir)
    
    if not model_to_evaluate:
        print("❌ Error: Could not find a model checkpoint to evaluate.")
        return
        
    # Construct the full path to the puzzle file
    puzzle_file_full_path = paths.local_project_root / args.puzzles


    run_evaluation(str(model_to_evaluate), str(puzzle_file_full_path), device)


if __name__ == '__main__':
    main()