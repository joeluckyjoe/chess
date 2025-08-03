import torch
import torch.optim as optim
import argparse
import sys
import chess
import chess.pgn
import datetime
from stockfish import Stockfish
from pathlib import Path
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm
from typing import Dict, List, TypedDict

# --- Import from config ---
from config import get_paths, config_params

# --- Project-specific Imports ---
from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.search.mcts import MCTS
from hardware_setup import get_device

# --- Type Hint for Game Memory ---
class GameStep(TypedDict):
    gnn_data: object
    cnn_tensor: torch.Tensor
    policy: torch.Tensor
    value_target: torch.Tensor

# --- Helper Functions ---
GNN_METADATA = (['square', 'piece'],[('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')])

def format_policy_for_training(policy_dict: Dict[chess.Move, float], board: chess.Board) -> torch.Tensor:
    """Converts the MCTS policy dictionary to a flat tensor for training."""
    policy_tensor = torch.zeros(get_action_space_size())
    for move, prob in policy_dict.items():
        action_index = move_to_index(move, board)
        if action_index is not None:
            policy_tensor[action_index] = prob
    return policy_tensor

def backfill_rewards(game_memory: List[Dict], final_result: float):
    """
    Backfills the final game result to every agent move in the game memory.
    This is the core of the pure RL signal for Phase C.
    """
    for step in game_memory:
        step['value_target'] = torch.tensor([final_result], dtype=torch.float32)

def train_on_game(model: PolicyValueModel, optimizer: optim.Optimizer, game_memory: List[GameStep], batch_size: int, device: torch.device):
    """Trains the model on the data collected from a single game."""
    if not game_memory:
        return 0.0, 0.0

    model.train()
    total_policy_loss, total_value_loss = 0, 0
    np.random.shuffle(game_memory) # Shuffle to break temporal correlations

    num_batches = (len(game_memory) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Training on game batches", leave=False):
        batch_memory = game_memory[i * batch_size : (i + 1) * batch_size]
        if not batch_memory:
            continue

        gnn_data_list = [item['gnn_data'] for item in batch_memory]
        cnn_tensors = torch.stack([item['cnn_tensor'] for item in batch_memory]).to(device)
        target_policies = torch.stack([item['policy'] for item in batch_memory]).to(device)
        target_values = torch.stack([item['value_target'] for item in batch_memory]).to(device)

        gnn_batch = Batch.from_data_list(gnn_data_list).to(device)

        optimizer.zero_grad()
        policy_logits, value_preds = model(gnn_batch, cnn_tensors)

        # Calculate losses
        policy_loss = torch.nn.functional.cross_entropy(policy_logits, target_policies)
        value_loss = torch.nn.functional.mse_loss(value_preds, target_values)
        total_loss = policy_loss + value_loss

        total_loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / num_batches, total_value_loss / num_batches

def save_checkpoint(model: PolicyValueModel, optimizer: optim.Optimizer, game_number: int, directory: Path):
    """Saves a training checkpoint."""
    # Updated checkpoint naming for Phase C
    checkpoint_path = directory / f"expert_sparring_checkpoint_game_{game_number}.pth.tar"
    torch.save({
        'game_number': game_number,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main():
    """Main execution function for the Phase C Expert Sparring training loop."""
    parser = argparse.ArgumentParser(description="Run the Phase C training loop against an expert opponent.")
    # Simplified argument for loading agent checkpoints
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a model checkpoint to continue training.")
    args = parser.parse_args()

    paths = get_paths()
    device = get_device()
    print(f"Using device: {device}")

    # --- Model and Optimizer Initialization ---
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
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_game = checkpoint.get('game_number', 0)
            print(f"Resuming from game {start_game + 1}")
        else:
            print(f"[WARNING] Checkpoint not found at {checkpoint_path}. Starting from scratch.")
    else:
        print("No checkpoint provided. Starting from scratch.")

    # --- MCTS and Stockfish Initialization ---
    mcts_player = MCTS(network=model, device=device, c_puct=config_params['CPUCT'], batch_size=config_params['BATCH_SIZE'])

    try:
        # Configure Stockfish to be a strong, consistent opponent
        stockfish_params = {
            # "depth" is a per-search parameter, not an initial one.
            "Skill Level": config_params.get('STOCKFISH_ELO', 1500)
        }
        stockfish_opponent = Stockfish(path=config_params['STOCKFISH_PATH'], parameters=stockfish_params)
        
        # We get the depth from config to use it in the game loop.
        stockfish_depth = config_params.get('STOCKFISH_DEPTH', 5) 
        print(f"Initialized Stockfish with Elo: {stockfish_params['Skill Level']} and Depth: {stockfish_depth}")

    except Exception as e:
        print(f"[FATAL] Could not initialize the Stockfish engine: {e}"); sys.exit(1)

    print("\n" + "#"*60 + "\n--- Phase C: Expert Sparring Training Begins ---\n" + "#"*60 + "\n")

    # --- Main Training Loop ---
    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        print(f"\n--- Starting Game {game_num}/{config_params['TOTAL_GAMES']} ---")
        board = chess.Board()
        game_memory: List[Dict] = []
        agent_is_white = (game_num % 2 == 1)
        print(f"Agent is playing as {'WHITE' if agent_is_white else 'BLACK'}")

        # PGN Game Setup
        game = chess.pgn.Game()
        game.headers["Event"] = "Phase C Expert Sparring"
        game.headers["Site"] = "Colab"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(game_num)
        game.headers["White"] = "Agent" if agent_is_white else f"Stockfish (Elo {stockfish_params['Skill Level']})"
        game.headers["Black"] = f"Stockfish (Elo {stockfish_params['Skill Level']})" if agent_is_white else "Agent"
        node = game

        while not board.is_game_over(claim_draw=True):
            is_agent_turn = (board.turn == chess.WHITE and agent_is_white) or (board.turn == chess.BLACK and not agent_is_white)

            if is_agent_turn:
                gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
                policy_dict = mcts_player.run_search(board, config_params['MCTS_SIMULATIONS'])

                if not policy_dict: break # Break if MCTS returns no moves

                # Select move and store state for later training
                move = mcts_player.select_move(policy_dict, temperature=1.0)
                policy_tensor = format_policy_for_training(policy_dict, board)
                game_memory.append({"gnn_data": gnn_data, "cnn_tensor": cnn_tensor, "policy": policy_tensor})

                san_move = board.san(move)
                print(f"Agent plays: {san_move}")
                board.push(move)
            else: # Stockfish's turn
                # Set the search depth before asking for the best move.
                stockfish_opponent.set_depth(stockfish_depth)
                
                stockfish_opponent.set_fen_position(board.fen())
                best_move_uci = stockfish_opponent.get_best_move()
                if not best_move_uci: break # Break if Stockfish returns no move

                move = chess.Move.from_uci(best_move_uci)
                san_move = board.san(move)
                print(f"Stockfish plays: {san_move}")
                board.push(move)

            node = node.add_variation(move)

        # --- Post-Game Processing ---
        result_str = board.result(claim_draw=True)
        game.headers["Result"] = result_str
        print(f"--- Game Over. Result: {result_str} ---")

        # Save PGN
        pgn_filename = paths.pgn_games_dir / f"expert_sparring_game_{game_num}.pgn"
        with open(pgn_filename, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)
        print(f"PGN for game saved to: {pgn_filename}")

        if game_memory:
            # Determine final reward based on game outcome
            final_reward = 0.0
            if (result_str == "1-0" and agent_is_white) or (result_str == "0-1" and not agent_is_white):
                final_reward = 1.0 # Agent won
            elif (result_str == "0-1" and agent_is_white) or (result_str == "1-0" and not agent_is_white):
                final_reward = -1.0 # Agent lost

            backfill_rewards(game_memory, final_reward)
            policy_loss, value_loss = train_on_game(model, optimizer, game_memory, config_params['BATCH_SIZE'], device)
            print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")

        # --- Checkpointing ---
        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            save_checkpoint(model, optimizer, game_num, paths.checkpoints_dir)

    print("\n--- Training Run Finished ---")

if __name__ == "__main__":
    main()