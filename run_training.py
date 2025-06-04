import os
import torch
from pathlib import Path

# --- NEW: Import the environment-aware path utility ---
from config import get_paths

# Core components from the gnn_agent package
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.search.mcts import MCTS
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer

def main():
    """
    Main training loop for the MCTS RL Chess Agent.
    Orchestrates self-play, data management, and network training.
    """
    # --- 1. Get Environment-Aware Paths ---
    # This will detect if we are in Colab, mount Google Drive if necessary,
    # and return the correct paths for data and checkpoints.
    checkpoints_path, training_data_path = get_paths()

    # --- 2. Configuration ---
    config = {
        # Training Run Parameters
        "total_games": 1000,
        "learning_rate": 0.001,
        "mcts_simulations": 50,
        "epochs_per_batch": 1,
        "temperature": 1.0,
        "temp_decay_moves": 30,

        # Checkpointing
        "save_checkpoint_every_n_games": 10,

        # Stockfish Engine - IMPORTANT: UPDATE THIS PATH IF NEEDED
        "stockfish_path": "/usr/games/stockfish",

        # Neural Network Architecture
        "gnn_input_features": 12,
        "gnn_hidden_features": 256,
        "gnn_output_features": 128,
        "attention_heads": 4,
        "policy_head_out_moves": 4672,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    # MODIFIED: Removed hardcoded "checkpoint_dir" and "training_data_dir"
    print(f"Using device: {config['device']}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")
    print(f"Training data will be saved to: {training_data_path}")

    # --- 3. Initialize All Components ---

    # Instantiate network sub-modules
    square_gnn = SquareGNN(in_features=config["gnn_input_features"], hidden_features=config["gnn_hidden_features"], out_features=config["gnn_output_features"], heads=config["attention_heads"])
    piece_gnn = PieceGNN(in_channels=config["gnn_input_features"], hidden_channels=config["gnn_hidden_features"], out_channels=config["gnn_output_features"])
    cross_attention = CrossAttentionModule(sq_embed_dim=config["gnn_output_features"], pc_embed_dim=config["gnn_output_features"], num_heads=config["attention_heads"])
    policy_head = PolicyHead(embedding_dim=config["gnn_output_features"], num_possible_moves=config["policy_head_out_moves"])
    value_head = ValueHead(embedding_dim=config["gnn_output_features"])

    # Instantiate the main network
    chess_network = ChessNetwork(square_gnn=square_gnn, piece_gnn=piece_gnn, cross_attention=cross_attention, policy_head=policy_head, value_head=value_head).to(config["device"])

    # Instantiate MCTS
    mcts = MCTS(network=chess_network, device=config["device"])

    # Instantiate SelfPlay with temperature parameters
    self_play = SelfPlay(
        mcts_white=mcts, 
        mcts_black=mcts, 
        stockfish_path=config["stockfish_path"], 
        num_simulations=config["mcts_simulations"],
        temperature=config["temperature"],
        temp_decay_moves=config["temp_decay_moves"]
    )

    # --- MODIFIED: Instantiate TrainingDataManager with the dynamic path ---
    training_data_manager = TrainingDataManager(
        data_directory=training_data_path
    )
    
    # Instantiate Trainer
    trainer = Trainer(network=chess_network, learning_rate=config["learning_rate"], device=config["device"])

    # --- 4. Load Checkpoint to Resume Training ---
    print("Attempting to load the latest checkpoint...")
    # --- MODIFIED: Use the dynamic path variable ---
    start_game = trainer.load_checkpoint(checkpoints_path)
    if start_game > 0:
        print(f"Resuming training from game {start_game + 1}")
    else:
        print("Starting training from scratch.")
        start_game = 0

    # --- 5. Main Training Loop ---
    for game_num in range(start_game + 1, config["total_games"] + 1):
        print(f"\n--- Starting Game {game_num} of {config['total_games']} ---")

        # a. Generate data via self-play
        print("Generating training data through self-play...")
        training_examples = self_play.play_game()
        if not training_examples:
            print("Game resulted in no training examples. Skipping.")
            continue
        print(f"Self-play complete. Generated {len(training_examples)} examples.")

        # b. Save the new data
        data_filename = f"game_{game_num}_data.pkl"
        training_data_manager.save_data(training_examples, filename=data_filename)

        # c. Train the network on the data from the game just played
        batch_data = training_examples
        
        print(f"Training on the {len(batch_data)} examples from game {game_num}...")
        for epoch in range(config["epochs_per_batch"]):
            print(f"Starting training epoch {epoch + 1}/{config['epochs_per_batch']}...")
            policy_loss, value_loss = trainer.train_on_batch(batch_data)
            print(f"Epoch {epoch + 1} complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")

        # d. Save a checkpoint periodically
        if game_num % config["save_checkpoint_every_n_games"] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            # --- MODIFIED: Use the dynamic path variable ---
            trainer.save_checkpoint(directory=checkpoints_path, game_number=game_num)

    print("\n--- Training Run Finished ---")


if __name__ == "__main__":
    main()