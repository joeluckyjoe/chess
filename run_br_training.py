import torch
import torch.optim as optim
import argparse
import sys
import chess
from stockfish import Stockfish
from pathlib import Path
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm
from typing import Dict

# --- Import from config ---
from config import get_paths, config_params

# --- Project-specific Imports ---
from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.rl_loop.style_classifier import StyleClassifier
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_action_index
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


def format_policy_for_training(policy_dict: Dict[chess.Move, float], board: chess.Board) -> torch.Tensor:
    """Converts the MCTS policy dictionary to a full-sized tensor for training."""
    policy_tensor = torch.zeros(get_action_space_size())
    for move, prob in policy_dict.items():
        action_index = move_to_action_index(move, board)
        if action_index is not None:
            policy_tensor[action_index] = prob
    return policy_tensor


def backfill_rewards(game_memory, final_result, discount_factor=0.95):
    """
    Backfills rewards with the discounted final game outcome.
    V(s_t) = r_t (style reward) + gamma * V(s_{t+1})
    """
    next_value = final_result
    for i in reversed(range(len(game_memory))):
        value_target = game_memory[i]['reward'] + next_value * discount_factor
        game_memory[i]['value_target'] = torch.tensor([value_target], dtype=torch.float32)
        next_value = value_target


def train_on_game(model, optimizer, game_memory, batch_size, device):
    """
    Performs a training step using the data collected from a single game.
    """
    if not game_memory:
        return 0.0, 0.0

    model.train()
    
    total_policy_loss = 0
    total_value_loss = 0
    
    np.random.shuffle(game_memory)
    
    num_batches = (len(game_memory) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Training on game batches"):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch_memory = game_memory[batch_start:batch_end]
        
        if not batch_memory:
            continue

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
    checkpoint_path = Path(directory) / f"br_checkpoint_game_{game_number}.pth.tar"
    torch.save({
        'game_number': game_number,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main execution function for the Phase BR training loop."""
    parser = argparse.ArgumentParser(description="Run the Phase BR training loop with style-based rewards.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a model checkpoint to continue training.")
    args = parser.parse_args()

    paths = get_paths()
    device = get_device()
    print(f"Using device: {device}")

    model = PolicyValueModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'],
        cnn_in_channels=14, 
        embed_dim=config_params['EMBED_DIM'],
        policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'],
        gnn_metadata=GNN_METADATA
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config_params['LEARNING_RATE'], weight_decay=config_params['WEIGHT_DECAY'])
    
    start_game = 0
    if args.checkpoint:
        # Load checkpoint logic...
        pass
    else:
        print("No checkpoint provided. Starting from scratch.")

    style_classifier = StyleClassifier()
    mcts_player = MCTS(network=model, device=device, c_puct=config_params['CPUCT'], batch_size=config_params['BATCH_SIZE'])
    
    try:
        stockfish_opponent = Stockfish(path=config_params['STOCKFISH_PATH'], depth=config_params.get('STOCKFISH_DEPTH_EVAL', 10))
    except Exception as e:
        print(f"[FATAL] Could not initialize the Stockfish engine: {e}"); sys.exit(1)

    print("\n" + "#"*60 + "\n--- Initialization Complete. Ready to start training. ---\n" + "#"*60 + "\n")

    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        print(f"\n--- Starting Game {game_num}/{config_params['TOTAL_GAMES']} ---")
        
        board = chess.Board()
        game_memory = []
        agent_is_white = (game_num % 2 == 1)
        print(f"Agent is playing as {'WHITE' if agent_is_white else 'BLACK'}")

        while not board.is_game_over(claim_draw=True):
            is_agent_turn = (board.turn == chess.WHITE and agent_is_white) or (board.turn == chess.BLACK and not agent_is_white)

            if is_agent_turn:
                gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
                policy_dict = mcts_player.run_search(board, config_params['MCTS_SIMULATIONS'])
                move = mcts_player.select_move(policy_dict, temperature=1.0)
                reward = style_classifier.score_move(board, move)
                
                # --- FIX: Convert policy dict to a full tensor ---
                policy_tensor = format_policy_for_training(policy_dict, board)
                
                game_memory.append({"gnn_data": gnn_data, "cnn_tensor": cnn_tensor, "policy": policy_tensor, "reward": reward})
                
                san_move = board.san(move); print(f"Agent plays: {san_move} (Style Reward: {reward:.3f})")
                board.push(move)
            else:
                stockfish_opponent.set_fen_position(board.fen())
                best_move_uci = stockfish_opponent.get_best_move()
                if best_move_uci:
                    move = chess.Move.from_uci(best_move_uci)
                    san_move = board.san(move); print(f"Stockfish plays: {san_move}")
                    board.push(move)
                else:
                    break
        
        result_str = board.result(claim_draw=True)
        print(f"--- Game Over. Result: {result_str} ---")

        if game_memory:
            if (result_str == "1-0" and agent_is_white) or (result_str == "0-1" and not agent_is_white):
                final_reward = 1.0
            elif (result_str == "0-1" and agent_is_white) or (result_str == "1-0" and not agent_is_white):
                final_reward = -1.0
            else:
                final_reward = 0.0
            
            backfill_rewards(game_memory, final_reward)
            policy_loss, value_loss = train_on_game(model, optimizer, game_memory, config_params['BATCH_SIZE'], device)
            print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            save_checkpoint(model, optimizer, game_num, paths.checkpoints_dir)

    print("\n--- Training Run Finished ---")

if __name__ == "__main__":
    main()