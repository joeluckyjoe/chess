import torch
from pathlib import Path
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Project imports (adjust paths if necessary)
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator
from gnn_agent.search.mcts import MCTS

class TrainingConfig:
    """Configuration parameters for the training loop."""
    # --- Paths ---
    # IMPORTANT: Update this path to your Stockfish executable
    STOCKFISH_PATH = "/usr/games/stockfish" 
    MODEL_SAVE_PATH = Path("./models/chess_agent_v3.5.pth")
    TRAINING_DATA_PATH = Path("./training_data/self_play_data.pkl")

    # --- RL Loop Parameters ---
    #NUM_ITERATIONS = 100         # Total number of training iterations (generation -> training)
    NUM_ITERATIONS = 1
    #NUM_SELF_PLAY_GAMES = 50     # Number of self-play games to generate per iteration
    NUM_SELF_PLAY_GAMES = 1
    #NUM_TRAINING_EPOCHS = 10     # Number of training epochs per iteration
    NUM_TRAINING_EPOCHS = 1

    # --- MCTS Parameters ---
    #NUM_MCTS_SIMULATIONS = 800   # MCTS simulations per move
    NUM_MCTS_SIMULATIONS = 16

    # --- Training Parameters ---
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4

def main():
    """
    The main driver for the reinforcement learning loop.
    """
    logging.info("--- Initializing Training Environment ---")

    # --- 1. Setup and Initialization (FINAL VERSION) ---
    config = TrainingConfig()

    config.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize the neural network
    model = ChessNetwork().to(device)

    if config.MODEL_SAVE_PATH.exists():
        logging.info(f"Loading existing model from {config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    # Initialize the data manager
    data_manager = TrainingDataManager(config.TRAINING_DATA_PATH)

    # Initialize the Trainer
    trainer = Trainer(network=model, learning_rate=config.LEARNING_RATE)

    # Create the MCTS agent, using the correct 'network' parameter
    mcts_agent = MCTS(network=model, device=device)

    # Create the SelfPlay manager, passing num_simulations to it
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
        
        # NEW a. Self-Play Phase
        all_training_examples = []
        for i in range(config.NUM_SELF_PLAY_GAMES):
            logging.info(f"  Running game {i + 1}/{config.NUM_SELF_PLAY_GAMES}...")
            
            # We assume the single-game method is run_game(). Let me know if it has a different name.
            game_data = self_play_manager.play_game(num_simulations=config.NUM_MCTS_SIMULATIONS)
            all_training_examples.extend(game_data)
        
        # b. Data Storage Phase
        logging.info(f"Saving {len(all_training_examples)} new training examples.")
        data_manager.save_data(all_training_examples, filename=config.TRAINING_DATA_PATH.name)

        # --- c. Training Phase (REWRITTEN) ---
        # The main script now handles epochs and batching.
        logging.info("Loading all training data for training phase.")
        all_training_data = data_manager.load_data(filename=config.TRAINING_DATA_PATH.name)
        
        if not all_training_data:
            logging.warning("No training data found. Skipping training for this iteration.")
            continue

        logging.info(f"Starting training: {len(all_training_data)} examples, {config.NUM_TRAINING_EPOCHS} epochs.")

        for epoch in range(config.NUM_TRAINING_EPOCHS):
            random.shuffle(all_training_data)
            logging.info(f"  Epoch {epoch + 1}/{config.NUM_TRAINING_EPOCHS}")
            
            progress_bar = 0 # Simple progress indicator
            for i in range(0, len(all_training_data), config.BATCH_SIZE):
                batch = all_training_data[i:i + config.BATCH_SIZE]
                if not batch:
                    continue
                
                # Call the low-level batch training method
                trainer.train_on_batch(batch)
                
                # Log progress
                progress_bar += len(batch)
                print(f"    Trained on {progress_bar}/{len(all_training_data)} examples...", end='\r')
            print() # Newline after progress bar

        # d. Save Model Checkpoint
        logging.info(f"Saving model checkpoint to {config.MODEL_SAVE_PATH}")
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    logging.info("\n--- Training Finished ---")
    self_play_manager.game.close() # Close the communicator instance used by SelfPlay

if __name__ == "__main__":
    main()