import torch
import chess
import numpy as np
from typing import Dict
import argparse
from pathlib import Path
import sys
import math

# --- Add project root to path for imports ---
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# --- Project-specific Imports ---
from gnn_agent.search.mcts import MCTS
from gnn_agent.search.mcts_node import MCTSNode
from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from config import config_params
from hardware_setup import get_device

# --- GNN Metadata ---
GNN_METADATA = (
    ['square', 'piece'],
    [('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'),
     ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')]
)


class VerboseMCTS(MCTS):
    """
    An MCTS subclass with added print statements for debugging and verification.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_counter = 0

    def _backpropagate(self, node: MCTSNode, value: float):
        print(f"    Backpropagating value: {value:.4f} from leaf node.")
        
        current_node = node
        current_value = value
        depth = 0
        while current_node:
            q_before = current_node.Q
            n_before = current_node.N
            
            current_node.N += 1
            current_node.Q += current_value
            
            print(f"      - Updating node at depth {depth}: N({n_before}->{current_node.N}), Q({q_before:.3f}->{current_node.Q:.3f}) with value {current_value:.3f}")

            if current_node.parent:
                print(f"        (Flipping value for parent: {current_value:.3f} -> {-current_value:.3f})")
                current_value *= -1

            current_node = current_node.parent
            depth += 1
        print("-" * 20)


    def run_search(self, board: chess.Board, num_simulations: int) -> Dict[chess.Move, float]:
        print("="*60)
        print("--- Starting New MCTS Debug Search ---")
        print(f"Position (FEN): {board.fen()}")
        print("="*60 + "\n")
        
        # --- Root Evaluation ---
        print("Step 1: Root Node Evaluation")
        super().run_search(board, 1)
        
        print(f"  - Root evaluated. Network value: {self.root.q_value():.4f}")
        print("  - Legal moves and their prior probabilities (P):")
        for move, child in sorted(self.root.children.items(), key=lambda item: item[1].P, reverse=True):
            print(f"    - {board.san(move)}: {child.P:.4f}")
        print("-" * 60 + "\n")

        # --- Simulation Loop ---
        print("Step 2: Running Simulations")
        for i in range(2, num_simulations + 1):
            print(f"\n--- Simulation {i}/{num_simulations} ---")
            
            # 1. Selection
            sim_board = board.copy()
            current_node = self.root
            path_str = "root"
            while not current_node.is_leaf():
                if not current_node.children: break
                
                print("  - Selecting child from node:")
                for m, n in current_node.children.items():
                    print(f"    - {sim_board.san(m)} -> UCT: {n.uct_value(self.c_puct):.4f} (Q: {-n.q_value():.3f}, P: {n.P:.3f}, N: {n.N})")

                best_move = max(current_node.children, key=lambda move: current_node.children[move].uct_value(self.c_puct))
                path_str += f" -> {sim_board.san(best_move)}"
                sim_board.push(best_move)
                current_node = current_node.children[best_move]
            print(f"  - Selection path: {path_str}")

            # 2. Expansion & Evaluation
            print("  - Reached leaf node. Evaluating position...")
            if not sim_board.is_game_over():
                self._pending_evaluations.append((current_node, sim_board))
                self._expand_and_evaluate_batch()
            else:
                # 3. Backpropagation for terminal node
                outcome = sim_board.outcome()
                term_value = 0.0
                if outcome and outcome.winner is not None:
                    player_at_leaf = current_node.board_turn_at_node
                    # --- FIX: Define the 'winner' variable ---
                    winner = outcome.winner
                    term_value = 1.0 if winner == player_at_leaf else -1.0
                print(f"  - Terminal node found. Game result value: {term_value}")
                self._backpropagate(current_node, term_value)

        # --- Final Policy Calculation ---
        if not self.root.children: return {}
        total_visits = sum(child.N for child in self.root.children.values())
        policy = {move: child.N / total_visits for move, child in self.root.children.items()} if total_visits > 0 else {}
        return policy

def load_model_for_debug(checkpoint_path, device):
    """Loads a PolicyValueModel from a checkpoint."""
    model = PolicyValueModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14, 
        embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path} and set to evaluation mode.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Run a verbose MCTS search for debugging.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument('--fen', type=str, required=True, help="FEN string of the board position to analyze.")
    parser.add_argument('--sims', type=int, default=10, help="Number of simulations to run.")
    args = parser.parse_args()

    device = get_device()
    model = load_model_for_debug(args.checkpoint, device)
    
    mcts_debugger = VerboseMCTS(network=model, device=device, batch_size=config_params['BATCH_SIZE'])
    
    board = chess.Board(args.fen)
    final_policy = mcts_debugger.run_search(board, args.sims)

    print("\n" + "="*60)
    print("--- MCTS Search Complete ---")
    print("Final policy based on visit counts:")
    print("-" * 60)
    print(f"{'Move':>10} | {'Visits (N)':>12} | {'Mean Q-Value':>14} | {'Prior (P)':>12}")
    print("-" * 60)
    
    sorted_children = sorted(mcts_debugger.root.children.items(), key=lambda item: item[1].N, reverse=True)

    for move, node in sorted_children:
        q_value_from_parent_perspective = -node.q_value()
        print(f"{board.san(move):>10} | {node.N:>12} | {q_value_from_parent_perspective:>14.4f} | {node.P:>12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()