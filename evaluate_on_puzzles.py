#
# File: evaluate_on_puzzles.py (Final Correction)
#
import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import chess

# --- Setup Python Path ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Project-specific Imports ---
from config import config_params
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.gamestate_converters import gnn_data_converter, action_space_converter


def load_model_via_trainer(checkpoint_path: str, device: torch.device) -> ChessNetwork:
    """
    Loads a ChessNetwork from a checkpoint using the canonical Trainer class.
    This ensures the model architecture is always consistent with the training script.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    trainer = Trainer(model_config=config_params, device=device)
    
    network, game_number = trainer.load_or_initialize_network(
        directory=None,
        specific_checkpoint_path=Path(checkpoint_path)
    )
    
    network.to(device)
    network.eval()
    
    print(f"✅ Successfully loaded model from game {game_number} for evaluation.")
    return network


def get_agent_top_move(board: chess.Board, network: ChessNetwork, device: torch.device) -> str:
    """
    Gets the agent's top-ranked move for a given board position.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    gnn_input = gnn_data_converter.convert_to_gnn_input(board, device)

    square_batch = torch.zeros(gnn_input.square_features.size(0), dtype=torch.long, device=device)
    piece_batch = torch.zeros(gnn_input.piece_features.size(0), dtype=torch.long, device=device)
    
    max_pieces = gnn_input.piece_features.size(0)
    piece_padding_mask = torch.ones((1, max_pieces), dtype=torch.bool, device=device)
    if max_pieces > 0:
        piece_padding_mask[0, :max_pieces] = 0

    with torch.no_grad():
        policy_logits, _ = network(
            square_features=gnn_input.square_features,
            square_edge_index=gnn_input.square_edge_index,
            square_batch=square_batch,
            piece_features=gnn_input.piece_features,
            piece_edge_index=gnn_input.piece_edge_index,
            piece_batch=piece_batch,
            piece_to_square_map=gnn_input.piece_to_square_map,
            piece_padding_mask=piece_padding_mask
        )

    policy_probs = torch.softmax(policy_logits.flatten(), dim=0)
    
    valid_legal_moves = []
    legal_move_indices_list = []
    
    for m in legal_moves:
        try:
            idx = action_space_converter.move_to_index(m, board)
            valid_legal_moves.append(m)
            legal_move_indices_list.append(idx)
        except ValueError:
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
    network = load_model_via_trainer(model_path, device)

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
        # --- FIX: Changed the key from 'best_move_uci' back to the correct 'best_move' ---
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
    print("           Puzzle Evaluation Complete")
    print("-------------------------------------------")
    print(f"Puzzles Solved: {puzzles_solved} / {total_puzzles}")
    print(f"Success Rate:    {success_rate:.2f}%")
    print("-------------------------------------------")


def main():
    """Parses arguments and launches the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a chess agent on tactical puzzles.")
    parser.add_argument("--model", type=str, required=True, help="Path to a specific model checkpoint.")
    parser.add_argument("--puzzles", type=str, required=True, help="Path to the puzzle file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = Path(args.model)
    puzzle_path = Path(args.puzzles)

    run_evaluation(str(model_path), str(puzzle_path), device)


if __name__ == '__main__':
    main()