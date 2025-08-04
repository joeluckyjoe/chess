import torch
import torch.nn as nn
import chess
import numpy as np
from typing import Dict, Optional, List
import argparse
from pathlib import Path
import sys
from collections import deque

# --- Add project root to path for imports ---
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# --- Project-specific Imports ---
from gnn_agent.search.mcts import MCTS
from gnn_agent.search.mcts_node import MCTSNode
from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.neural_network.temporal_model import TemporalPolicyValueModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size
from config import config_params
from hardware_setup import get_device

# --- GNN Metadata ---
GNN_METADATA = (
    ['square', 'piece'],
    [('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'),
     ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')]
)
SEQUENCE_LENGTH = 8

class VerboseMCTS(MCTS):
    """ An MCTS subclass with added print statements for debugging. """
    def run_search(self, board: chess.Board, num_simulations: int, state_sequence: Optional[List[tuple]] = None) -> Dict[chess.Move, float]:
        print("="*60)
        print("--- Starting New MCTS Debug Search ---")
        print(f"Position (FEN): {board.fen()}")
        print(f"Model Type: {'Temporal' if self.is_temporal else 'GNN'}")
        print("="*60 + "\n")

        # --- Root Evaluation (Single Simulation) ---
        print("Step 1: Root Node Evaluation")
        super().run_search(board, 1, state_sequence)
        
        print(f"  - Root evaluated. Network value: {self.root.q_value():.4f}")
        print("  - Legal moves and their prior probabilities (P):")
        if not self.root.children:
            print("  - No legal moves from this position.")
        else:
            for move, child in sorted(self.root.children.items(), key=lambda item: item[1].P, reverse=True):
                print(f"    - {board.san(move)}: {child.P:.4f}")
        print("-" * 60 + "\n")

        # --- Simulation Loop ---
        print(f"Step 2: Running remaining {num_simulations - 1} Simulations")
        # We've already done 1 simulation for the root, so we do n-1 more.
        if num_simulations > 1:
            # The superclass method handles the loop internally
            super().run_search(board, num_simulations, state_sequence)

        # --- Final Policy Calculation ---
        if not self.root.children: return {}
        total_visits = sum(child.N for child in self.root.children.values())
        policy = {move: child.N / total_visits for move, child in self.root.children.items()} if total_visits > 0 else {}
        return policy

def load_model_for_debug(checkpoint_path_str: str, device: torch.device) -> nn.Module:
    """Loads the appropriate model (Temporal or GNN) from a checkpoint."""
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    
    is_temporal = any(key.startswith('transformer_encoder') for key in model_state_dict.keys())

    if is_temporal:
        print("Detected Temporal Model.")
        encoder_model = PolicyValueModel(
            gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
            embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
            gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
        )
        model = TemporalPolicyValueModel(
            encoder_model=encoder_model,
            policy_size=get_action_space_size(),
            d_model=config_params['EMBED_DIM']
        ).to(device)
    else:
        print("Detected GNN Model.")
        model = PolicyValueModel(
            gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
            embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
            gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
        ).to(device)
        
    model.load_state_dict(model_state_dict)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Run a verbose MCTS search for debugging.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument('--fen', type=str, required=True, help="FEN string of the board position to analyze.")
    parser.add_argument('--sims', type=int, default=50, help="Number of simulations to run.")
    args = parser.parse_args()

    device = get_device()
    model = load_model_for_debug(args.checkpoint, device)
    
    mcts_debugger = VerboseMCTS(network=model, device=device, batch_size=config_params['BATCH_SIZE'])
    
    board = chess.Board(args.fen)
    
    state_sequence = None
    if mcts_debugger.is_temporal:
        # Create a dummy state sequence for single-position analysis
        initial_gnn, initial_cnn, _ = convert_to_gnn_input(chess.Board(), device)
        state_sequence_queue = deque([(initial_gnn, initial_cnn)] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
        state_sequence = list(state_sequence_queue)

    final_policy = mcts_debugger.run_search(board, args.sims, state_sequence)

    print("\n" + "="*60)
    print("--- MCTS Search Complete ---")
    print("Final policy based on visit counts:")
    print("-" * 60)
    print(f"{'Move':>10} | {'Visits (N)':>12} | {'Mean Q-Value':>14} | {'Prior (P)':>12}")
    print("-" * 60)
    
    if not mcts_debugger.root.children:
        print("No moves to display.")
    else:
        sorted_children = sorted(mcts_debugger.root.children.items(), key=lambda item: item[1].N, reverse=True)
        for move, node in sorted_children:
            q_value_from_parent_perspective = -node.q_value()
            print(f"{board.san(move):>10} | {node.N:>12} | {q_value_from_parent_perspective:>14.4f} | {node.P:>12.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()