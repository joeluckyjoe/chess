import os
import torch
from pathlib import Path

# --- MODIFIED: Import both config_params and get_paths from config ---
from config import get_paths, config_params

# Core components from the gnn_agent package
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gn_agent.neural_network.chess_network import ChessNetwork
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
    checkpoints_path, training_data_path = get_paths()
    
    # --- 2. Configuration is now loaded from config.py ---
    # The local 'config' dictionary has been removed. We now use 'config_params'.
    device = config_params['DEVICE']
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")
    print(f"Training data will be saved to: {training_data_path}")

    # --- 3. Initialize All Components (using config_params) ---

    # Instantiate network sub-modules based on architecture params in config
    square_gnn = SquareGNN(
        in_features=12,  # Hardcoded for now, standard for our setup
        hidden_features=256,
        out_features=128,
        heads=4
    )
    piece_gnn = PieceGNN(
        in_channels=12, # Hardcoded for now
        hidden_channels=256,
        out_channels=128
    )
    cross_attention = CrossAttentionModule(
        sq_embed_dim=128,
        pc_embed_dim=128,
        num_heads=4
    )
    policy_head = PolicyHead(
        embedding_dim=128,
        num_possible_moves=4672 # Hardcoded for now
    )
    value_head = ValueHead(embedding_dim=128)

    # Instantiate the main network
    chess_network = ChessNetwork(
        square_gnn=square_gnn, 
        piece_gnn=piece_gnn, 
        cross_attention=cross_attention, 
        policy_head=policy_head, 
        value_head=value_head
    ).to(device)

    # Instantiate MCTS
    # --- MODIFIED: Changed 'cpuct' to 'c_puct' to match the MCTS class constructor ---
    mcts = MCTS(network=chess_network, device=device, c_puct=config_params['CPUCT'])

    # Instantiate SelfPlay
    self_play = SelfPlay(
        mcts_white=mcts,
        mcts_black=mcts,
        stockfish_path=config_params['STOCKFISH_PATH'],
        num_simulations=config_params['MCTS_SIMULATIONS'],
        temperature=1.0, # Initial temperature, can be made configurable if needed
        temp_decay_moves=30, # Can be made configurable
        print_move_timers=False # Can be made configurable
    )

    # Instantiate TrainingDataManager
    training_data_manager = TrainingDataManager(
        data_directory=training_data_path
    )
    
    # Instantiate Trainer, passing the full config_params for checkpointing
    trainer = Trainer(
        network=chess_network,
        model_config=config_params, 
        learning_rate=config_params['LEARNING_RATE'],
        weight_decay=config_params['WEIGHT_DECAY'],
        device=device
    )

    # --- 4. Load Checkpoint to Resume Training ---
    print("Attempting to load the latest checkpoint...")
    start_game = trainer.load_checkpoint(checkpoints_path)
    if start_game > 0:
        print(f"Resuming training from game {start_game + 1}")
    else:
        print("Starting training from scratch.")
        start_game = 0

    # --- 5. Main Training Loop ---
    for game_num in range(start_game + 1, config_params['NUM_SELF_PLAY_GAMES'] + 1):
        print(f"\n--- Starting Game {game_num} of {config_params['NUM_SELF_PLAY_GAMES']} ---")

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
        for epoch in range(config_params['TRAINING_EPOCHS']):
            print(f"Starting training epoch {epoch + 1}/{config_params['TRAINING_EPOCHS']}...")
            policy_loss, value_loss = trainer.train_on_batch(batch_data, batch_size=config_params['BATCH_SIZE'])
            print(f"Epoch {epoch + 1} complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")

        # d. Save a checkpoint periodically
        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=checkpoints_path, game_number=game_num)

    print("\n--- Training Run Finished ---")


if __name__ == "__main__":
    main()