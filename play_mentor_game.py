import torch
import chess
import random
import pickle
from pathlib import Path
import time
import argparse
import sys

# --- Import from config ---
# This is now guaranteed to work because we are creating the necessary __init__.py files
from config import get_paths, config_params

# --- Core component imports, mirroring evaluate_agent.py and the project structure ---
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

# =================================================================
# Helper Functions (Adapted from evaluate_agent.py for consistency)
# =================================================================

def load_agent_from_checkpoint(checkpoint_path: Path, device: str) -> tuple[ChessNetwork, dict]:
    """
    Loads the ChessNetwork model from a checkpoint by reconstructing its architecture.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
    print(f"Loading agent network from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Use the config from the checkpoint to ensure model architecture is identical
    cfg = checkpoint.get('config_params')
    if cfg is None:
        raise ValueError("Critical Error: 'config_params' not found in checkpoint.")

    print("Reconstructing model from saved config_params...")
    # NOTE: These parameters should ideally also come from the config file,
    # but for now we hardcode them as in the evaluation script for consistency.
    square_gnn = SquareGNN(in_features=12, hidden_features=256, out_features=128, heads=4)
    piece_gnn = PieceGNN(in_channels=12, hidden_channels=256, out_channels=128)
    cross_attention = CrossAttentionModule(sq_embed_dim=128, pc_embed_dim=128, num_heads=4)
    policy_head = PolicyHead(embedding_dim=128, num_possible_moves=4672)
    value_head = ValueHead(embedding_dim=128)

    model = ChessNetwork(
        square_gnn=square_gnn,
        piece_gnn=piece_gnn,
        cross_attention=cross_attention,
        policy_head=policy_head,
        value_head=value_head
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Agent network loaded and reconstructed successfully.")
    return model, cfg

def initialize_stockfish_player(stockfish_exe_path: str) -> StockfishCommunicator:
    """Initializes StockfishCommunicator using the project's custom class."""
    if not stockfish_exe_path or not Path(stockfish_exe_path).is_file():
        raise FileNotFoundError(f"Stockfish executable not found at '{stockfish_exe_path}'")

    print(f"Initializing Stockfish via StockfishCommunicator (path: {stockfish_exe_path})...")
    sf_comm = StockfishCommunicator(stockfish_path=stockfish_exe_path)
    if not sf_comm.perform_handshake():
        raise RuntimeError("Stockfish UCI handshake failed.")
    print("Stockfish initialized and handshake successful.")
    return sf_comm

def get_stockfish_move(sf_comm: StockfishCommunicator, board_fen: str, depth: int) -> str:
    """Gets Stockfish's best move using the project's custom class."""
    # This logic is copied directly from evaluate_agent.py
    timeout_sec = 20 
    sf_comm._send_command(f"position fen {board_fen}")
    isready_success, _ = sf_comm._raw_uci_command_exchange("isready", "readyok", timeout=timeout_sec)
    if not isready_success:
        raise RuntimeError("Stockfish not ready after setting position.")

    sf_comm._send_command(f"go depth {depth}")
    token_found, lines = sf_comm._read_output_until("bestmove", timeout=timeout_sec)
    
    if token_found:
        for line in reversed(lines):
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    raise RuntimeError(f"Stockfish did not return a 'bestmove' within {timeout_sec}s for depth {depth}.")

def assign_colors(agent_color_config: str):
    """Assigns colors to agent and mentor based on config."""
    if agent_color_config == "random":
        return random.choice([(chess.WHITE, chess.BLACK), (chess.BLACK, chess.WHITE)])
    elif agent_color_config == "white":
        return chess.WHITE, chess.BLACK
    else: # black
        return chess.BLACK, chess.WHITE

# =================================================================
# Main Game Logic
# =================================================================

def play_mentor_game(mcts_player, agent_color, stockfish_player, device):
    """
    Plays a single game between the MCTS agent and the Stockfish mentor,
    collecting training data from the agent's perspective.
    """
    board = chess.Board()
    training_examples = []
    
    stockfish_depth = config_params['MENTOR_STOCKFISH_DEPTH']
    mcts_sims = config_params['MCTS_SIMULATIONS']

    while not board.is_game_over(claim_draw=True):
        is_mcts_turn = board.turn == agent_color

        if is_mcts_turn:
            print("\nAgent's turn...")
            # Run MCTS search to get policy (pi) and best move
            pi, best_move_obj, _ = mcts_player.run_search(board, mcts_sims)
            
            # Store the state (FEN) and policy vector for training
            # The result (z) will be filled in at the end of the game
            training_examples.append([board.fen(), pi, None]) 
            
            move_uci = best_move_obj.uci()
            print(f"Agent plays: {move_uci}")
            board.push(best_move_obj)

        else: # Stockfish's turn
            print("\nMentor's turn...")
            move_uci = get_stockfish_move(stockfish_player, board.fen(), stockfish_depth)
            print(f"Mentor plays: {move_uci}")
            board.push_uci(move_uci)
        
        print(f"Board FEN: {board.fen()}")
        print(board)

    # --- Game Over ---
    result_str = board.result(claim_draw=True)
    if result_str == "1-0":
        agent_perspective_result = 1.0 if agent_color == chess.WHITE else -1.0
    elif result_str == "0-1":
        agent_perspective_result = -1.0 if agent_color == chess.WHITE else 1.0
    else: # Draw
        agent_perspective_result = 0.0
    
    print(f"\nGame Over. Result: {result_str} -> Agent gets {agent_perspective_result}")

    # Backpropagate the final game result to all stored training examples
    for example in training_examples:
        example[2] = agent_perspective_result
        
    return training_examples

# =================================================================
# Script Entry Point
# =================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a single mentor game and save the data.")
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        required=True,
        help="Filename of the agent's checkpoint file (must be in the checkpoints directory)."
    )
    args = parser.parse_args()

    # --- Setup ---
    print("--- Initializing Mentor Game ---")
    checkpoints_path, training_data_path = get_paths()
    
    device_str = config_params['DEVICE']
    device = "cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu"
    print(f"Using device: {device}")
    
    stockfish_player = None
    try:
        # --- Load Model ---
        model_checkpoint_path = checkpoints_path / args.checkpoint
        agent_model, loaded_config = load_agent_from_checkpoint(model_checkpoint_path, device)

        # --- Initialize Players ---
        # Use the c_puct value from the main config file
        mcts_player = MCTS(network=agent_model, device=device, c_puct=config_params['CPUCT'])
        
        # Use the stockfish path from the main config file
        stockfish_path = config_params['STOCKFISH_PATH']
        stockfish_player = initialize_stockfish_player(stockfish_path)

        # --- Assign Colors ---
        agent_color, mentor_color = assign_colors(config_params['MENTOR_GAME_AGENT_COLOR'])
        print(f"Agent plays as {chess.COLOR_NAMES[agent_color]}, Mentor as {chess.COLOR_NAMES[mentor_color]}")

        # --- Play Game ---
        training_data = play_mentor_game(
            mcts_player=mcts_player,
            agent_color=agent_color,
            stockfish_player=stockfish_player,
            device=device
        )
        
        # --- Save Data ---
        if training_data:
            # Using a timestamp ensures unique filenames for each game
            filename = f"mentor_game_{int(time.time())}.pkl"
            save_path = training_data_path / filename
            with open(save_path, 'wb') as f:
                pickle.dump(training_data, f)
            print(f"\nSuccessfully saved {len(training_data)} training examples to: {save_path}")
        else:
            print("\nNo training data was generated (agent might not have had a turn).")

    except Exception as e:
        print(f"\nFATAL ERROR during mentor game: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        if stockfish_player:
            stockfish_player.close()
            print("Stockfish communicator closed.")
        print("Mentor game script finished.")

