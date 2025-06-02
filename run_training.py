import torch
from pathlib import Path
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Project imports
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator
from gnn_agent.search.mcts import MCTS

class TrainingConfig:
    """Configuration parameters for the training loop."""
    # --- Paths ---
    STOCKFISH_PATH = "/usr/games/stockfish"
    MODEL_SAVE_PATH = Path("./models/chess_agent_v3.8.pth")
    TRAINING_DATA_PATH = Path("./training_data/self_play_data.pkl")

    # --- RL Loop Parameters ---
    NUM_ITERATIONS = 1
    NUM_SELF_PLAY_GAMES = 1
    NUM_TRAINING_EPOCHS = 1

    # --- MCTS Parameters ---
    NUM_MCTS_SIMULATIONS = 16

    # --- Training Parameters ---
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4

    # --- Network Parameters (CORRECTED) ---
    GNN_INPUT_FEATURES = 12     # CORRECTED: Changed from 8 to 12 to match the data converter
    GNN_HIDDEN_DIM = 128
    GNN_OUTPUT_DIM = 256
    SQUARE_GNN_HEADS = 4
    ATTENTION_HEADS = 8
    ATTENTION_DROPOUT = 0.1
    POLICY_OUTPUT_SIZE = 4672
    
def main():
    """
    The main driver for the reinforcement learning loop.
    """
    logging.info("--- Initializing Training Environment ---")

    # --- 1. Setup and Initialization ---
    config = TrainingConfig()

    config.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Initialize Neural Network Components ---
    logging.info("Instantiating network components with correct parameters...")
    
    square_gnn = SquareGNN(
        in_features=config.GNN_INPUT_FEATURES,
        hidden_features=config.GNN_HIDDEN_DIM,
        out_features=config.GNN_OUTPUT_DIM,
        heads=config.SQUARE_GNN_HEADS
    )

    piece_gnn = PieceGNN(
        in_channels=config.GNN_INPUT_FEATURES,
        hidden_channels=config.GNN_HIDDEN_DIM,
        out_channels=config.GNN_OUTPUT_DIM
    )
    
    cross_attention_module = CrossAttentionModule(
        sq_embed_dim=config.GNN_OUTPUT_DIM,
        pc_embed_dim=config.GNN_OUTPUT_DIM,
        num_heads=config.ATTENTION_HEADS,
        dropout_rate=config.ATTENTION_DROPOUT
    )

    policy_head = PolicyHead(
        embedding_dim=config.GNN_OUTPUT_DIM,
        num_possible_moves=config.POLICY_OUTPUT_SIZE
    )
    
    value_head = ValueHead(
        embedding_dim=config.GNN_OUTPUT_DIM
    )

    # --- Assemble the Full Network (CORRECTED) ---
    model = ChessNetwork(
        square_gnn=square_gnn,
        piece_gnn=piece_gnn,
        cross_attention=cross_attention_module, # Corrected keyword from 'attention_module'
        policy_head=policy_head,
        value_head=value_head
    ).to(device)

    logging.info("ChessNetwork instantiated successfully.")

    if config.MODEL_SAVE_PATH.exists():
        logging.info(f"Loading existing model from {config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    # Initialize other components
    data_manager = TrainingDataManager(config.TRAINING_DATA_PATH)
    trainer = Trainer(network=model, learning_rate=config.LEARNING_RATE)
    mcts_agent = MCTS(network=model, device=device)
    self_play_manager = SelfPlay(
        mcts_white=mcts_agent,
        mcts_black=mcts_agent,
        stockfish_path=config.STOCKFISH_PATH,
        num_simulations=config.NUM_MCTS_SIMULATIONS
    )

    logging.info("--- Initialization Complete ---")

    # --- 2. Main Training Loop ---
    for iteration in range(1, config.NUM_ITERATIONS + 1):
        logging.info(f"\n{'='*20} Starting Iteration {iteration}/{config.NUM_ITERATIONS} {'='*20}")

        # a. Self-Play Phase
        logging.info(f"Starting self-play phase: generating {config.NUM_SELF_PLAY_GAMES} games.")
        data_manager.clear_data() 
                
        all_training_examples = []
        for i in range(config.NUM_SELF_PLAY_GAMES):
            logging.info(f"  Running game {i + 1}/{config.NUM_SELF_PLAY_GAMES}...")
            # CORRECTED: Pass the number of simulations to the method call
            game_data = self_play_manager.play_game(num_simulations=config.NUM_MCTS_SIMULATIONS)
            all_training_examples.extend(game_data)
        
        # b. Data Storage Phase
        logging.info(f"Saving {len(all_training_examples)} new training examples.")
        data_manager.save_data(all_training_examples, filename=config.TRAINING_DATA_PATH.name)

        # --- c. Training Phase ---
        logging.info("Loading all training data for training phase.")
        all_training_data = data_manager.load_data(filename=config.TRAINING_DATA_PATH.name)
        
        if not all_training_data:
            logging.warning("No training data found. Skipping training for this iteration.")
            continue

        logging.info(f"Starting training: {len(all_training_data)} examples, {config.NUM_TRAINING_EPOCHS} epochs.")

        for epoch in range(config.NUM_TRAINING_EPOCHS):
            random.shuffle(all_training_data)
            logging.info(f"  Epoch {epoch + 1}/{config.NUM_TRAINING_EPOCHS}")
            
            progress_bar = 0
            for i in range(0, len(all_training_data), config.BATCH_SIZE):
                batch = all_training_data[i:i + config.BATCH_SIZE]
                if not batch:
                    continue
                
                trainer.train_on_batch(batch)
                
                progress_bar += len(batch)
                print(f"    Trained on {progress_bar}/{len(all_training_data)} examples...", end='\r')
            print()

        # d. Save Model Checkpoint
        logging.info(f"Saving model checkpoint to {config.MODEL_SAVE_PATH}")
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    logging.info("\n--- Training Finished ---")
    self_play_manager.game.close()

if __name__ == "__main__":
    main()