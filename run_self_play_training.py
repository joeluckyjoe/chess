import torch
import torch.optim as optim
import argparse
import sys
import chess
import chess.pgn
import datetime
from pathlib import Path
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm
from typing import Dict, List, Tuple

# --- Import from config ---
from config import get_paths, config_params

# --- Project-specific Imports ---
from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.search.mcts import MCTS
from hardware_setup import get_device

# --- GNN Metadata (single source of truth) ---
GNN_METADATA = (
    ['square', 'piece'],
    [
        ('square', 'adjacent_to', 'square'),
        ('piece', 'occupies', 'square'),
        ('piece', 'attacks', 'piece'),
        ('piece', 'defends', 'piece')
    ]
)

class SelfPlay:
    """
    Manages the process of a single self-play game.
    """
    def __init__(self, model: PolicyValueModel, device: torch.device):
        self.model = model
        self.device = device
        self.mcts = MCTS(network=self.model, device=self.device,
                         c_puct=config_params['CPUCT'],
                         batch_size=config_params['BATCH_SIZE'])

    def play_game(self) -> Tuple[List[Dict], float]:
        game_memory = []
        board = chess.Board()
        
        while not board.is_game_over(claim_draw=True):
            gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, self.device)
            policy_dict = self.mcts.run_search(board, config_params['MCTS_SIMULATIONS'])
            
            # For training, use temperature to encourage exploration
            move = self.mcts.select_move(policy_dict, temperature=1.0)
            
            # Store the state and the MCTS policy for later training
            policy_tensor = torch.zeros(get_action_space_size())
            for m, p in policy_dict.items():
                policy_tensor[move_to_index(m, board)] = p
            
            game_memory.append({
                "gnn_data": gnn_data,
                "cnn_tensor": cnn_tensor,
                "policy": policy_tensor,
            })
            
            board.push(move)

        result_str = board.result(claim_draw=True)
        if result_str == "1-0":
            final_reward = 1.0
        elif result_str == "0-1":
            final_reward = -1.0
        else:
            final_reward = 0.0
            
        return game_memory, final_reward, chess.pgn.Game.from_board(board)

def backfill_rewards(game_memory: List[Dict], final_reward: float, discount_factor: float = 0.98):
    """
    Backfills the rewards from the end of the game.
    The reward for a state is the discounted result from the perspective of the player to move.
    """
    reward = final_reward
    for i in reversed(range(len(game_memory))):
        game_memory[i]['value_target'] = torch.tensor([reward], dtype=torch.float32)
        # Flip the reward for the opponent's turn
        reward *= -discount_factor

def train_on_game(model, optimizer, game_memory, batch_size, device):
    """
    Performs a training step using the data collected from a single game.
    """
    if not game_memory: return 0.0, 0.0
    model.train()
    
    np.random.shuffle(game_memory)
    total_policy_loss, total_value_loss = 0.0, 0.0
    num_batches = (len(game_memory) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        batch_memory = game_memory[i * batch_size : (i + 1) * batch_size]
        if not batch_memory: continue

        gnn_data_list = [item['gnn_data'] for item in batch_memory]
        cnn_tensors = torch.stack([item['cnn_tensor'] for item in batch_memory]).to(device)
        target_policies = torch.stack([item['policy'] for item in batch_memory]).to(device)
        target_values = torch.stack([item['value_target'] for item in batch_memory]).to(device)
        
        gnn_batch = Batch.from_data_list(gnn_data_list).to(device)
        
        optimizer.zero_grad()
        policy_logits, value_preds = model(gnn_batch, cnn_tensors)

        policy_loss = torch.nn.functional.cross_entropy(policy_logits, target_policies)
        value_loss = torch.nn.functional.mse_loss(value_preds, target_values)
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
    return total_policy_loss / num_batches, total_value_loss / num_batches

def save_checkpoint(model, optimizer, game_number, directory):
    """Saves a training checkpoint."""
    checkpoint_path = Path(directory) / f"selfplay_checkpoint_game_{game_number}.pth.tar"
    torch.save({
        'game_number': game_number,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="Run the pure self-play RL training loop (Phase C).")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a model checkpoint to continue training.")
    args = parser.parse_args()

    paths = get_paths()
    device = get_device()
    print(f"Using device: {device}")

    model = PolicyValueModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14, 
        embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config_params['LEARNING_RATE'], weight_decay=config_params['WEIGHT_DECAY'])
    
    start_game = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading agent checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_game = checkpoint.get('game_number', 0)
            print(f"Resuming from game {start_game + 1}")
    else:
        print("No agent checkpoint provided. Starting a new run from scratch.")

    self_play_worker = SelfPlay(model, device)
    
    print("\n" + "#"*60 + "\n--- Initialization Complete. Starting Pure Self-Play Training. ---\n" + "#"*60 + "\n")

    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        print(f"\n--- Starting Self-Play Game {game_num}/{config_params['TOTAL_GAMES']} ---")
        
        game_memory, final_reward, pgn_data = self_play_worker.play_game()
        
        result_str = pgn_data.headers.get("Result", "*")
        print(f"--- Game Over. Result: {result_str}. Moves: {len(game_memory)} ---")

        pgn_filename = paths.pgn_games_dir / f"selfplay_game_{game_num}.pgn"
        with open(pgn_filename, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            pgn_data.accept(exporter)
            
        backfill_rewards(game_memory, final_reward)
        policy_loss, value_loss = train_on_game(model, optimizer, game_memory, config_params['BATCH_SIZE'], device)
        print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            save_checkpoint(model, optimizer, game_num, paths.checkpoints_dir)

    print("\n--- Training Run Finished ---")

if __name__ == "__main__":
    main()