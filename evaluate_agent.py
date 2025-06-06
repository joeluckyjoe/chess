# evaluate_agent.py (Final, Refactored Version)

import torch
import chess
import chess.pgn
import datetime
import os
from pathlib import Path
import time
import argparse # <<< NEW: Import library for command-line arguments

# --- Module Imports ---
from config import get_paths
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

# --- Configuration ---
CHECKPOINTS_DIR, DATA_DIR = get_paths()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Evaluation Parameters ---
# <<< REMOVED: The hardcoded MODEL_CHECKPOINT_FILENAME is now a command-line argument >>>

MCTS_SIMULATIONS_PER_MOVE = 100      # For our agent
STOCKFISH_GO_COMMAND_DEPTH = 5      # Using "go depth X" for Stockfish
STOCKFISH_TIMEOUT_SECONDS = 20      # Timeout for Stockfish to find a move
NUM_EVALUATION_GAMES = 20           # Total games (will be split for colors)

# --- Helper Functions ---

def load_agent_from_checkpoint(checkpoint_path: Path) -> tuple[ChessNetwork, dict]:
    """
    Loads the ChessNetwork model from a checkpoint by first reconstructing
    its architecture from the saved configuration parameters.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
    print(f"Loading agent network from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    cfg = checkpoint.get('config_params')
    if cfg is None:
        raise ValueError("Critical Error: 'config_params' not found in checkpoint. "
                         "Cannot reconstruct model architecture. Please use a checkpoint "
                         "saved with the updated Trainer.")

    print("Reconstructing model from saved config_params...")
    
    # Instantiate sub-modules using parameters from the checkpoint's config
    square_gnn = SquareGNN(
        in_features=cfg["gnn_input_features"],
        hidden_features=cfg["gnn_hidden_features"],
        out_features=cfg["gnn_output_features"],
        heads=cfg["attention_heads"]
    )
    piece_gnn = PieceGNN(
        in_channels=cfg["gnn_input_features"],
        hidden_channels=cfg["gnn_hidden_features"],
        out_channels=cfg["gnn_output_features"]
    )
    cross_attention = CrossAttentionModule(
        sq_embed_dim=cfg["gnn_output_features"],
        pc_embed_dim=cfg["gnn_output_features"],
        num_heads=cfg["attention_heads"]
    )
    policy_head = PolicyHead(
        embedding_dim=cfg["gnn_output_features"],
        num_possible_moves=cfg["policy_head_out_moves"]
    )
    value_head = ValueHead(embedding_dim=cfg["gnn_output_features"])

    # Instantiate the main network with the reconstructed sub-modules
    model = ChessNetwork(
        square_gnn=square_gnn,
        piece_gnn=piece_gnn,
        cross_attention=cross_attention,
        policy_head=policy_head,
        value_head=value_head
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval() # Set to evaluation mode
    print("Agent network loaded and reconstructed successfully.")
    return model, cfg

def initialize_stockfish_player(stockfish_exe_path: str) -> StockfishCommunicator:
    """Initializes StockfishCommunicator and performs handshake."""
    if not stockfish_exe_path or not Path(stockfish_exe_path).is_file():
           raise FileNotFoundError(f"Stockfish executable not found at '{stockfish_exe_path}'")

    print(f"Initializing Stockfish (path: {stockfish_exe_path})...")
    sf_comm = StockfishCommunicator(stockfish_path=stockfish_exe_path)
    if not sf_comm.perform_handshake():
        raise RuntimeError("Stockfish UCI handshake failed.")
    print("Stockfish initialized and handshake successful.")
    return sf_comm

def get_stockfish_move(sf_comm: StockfishCommunicator, board_fen: str, depth: int, timeout_sec: int) -> str:
    """Gets Stockfish's best move for a given FEN and depth."""
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

def play_evaluation_game(our_agent_color_is_white, mcts_player, stockfish_player, game_board, pgn_game_handler, stockfish_path):
    """Plays a single game and returns the outcome from our agent's perspective."""
    game_board.reset()
    pgn_game_handler.headers["White"] = "MCTS_Agent" if our_agent_color_is_white else f"Stockfish_D{STOCKFISH_GO_COMMAND_DEPTH}"
    pgn_game_handler.headers["Black"] = f"Stockfish_D{STOCKFISH_GO_COMMAND_DEPTH}" if our_agent_color_is_white else "MCTS_Agent"
    
    current_pgn_node = pgn_game_handler

    while not game_board.is_game_over(claim_draw=True):
        is_mcts_turn = (game_board.turn == chess.WHITE and our_agent_color_is_white) or \
                         (game_board.turn == chess.BLACK and not our_agent_color_is_white)
        
        player_name = "MCTS Agent" if is_mcts_turn else f"Stockfish_D{STOCKFISH_GO_COMMAND_DEPTH}"
        
        move_uci = None
        if is_mcts_turn:
            _, best_move_obj, _ = mcts_player.run_search(game_board, MCTS_SIMULATIONS_PER_MOVE)
            if best_move_obj:
                move_uci = best_move_obj.uci()
        else: # Stockfish's turn
            move_uci = get_stockfish_move(stockfish_player, game_board.fen(), STOCKFISH_GO_COMMAND_DEPTH, STOCKFISH_TIMEOUT_SECONDS)

        if not move_uci:
            print(f"ERROR: {player_name} failed to produce a move string. Ending game.")
            return 1 if player_name.startswith("Stockfish") else -1

        move_obj = chess.Move.from_uci(move_uci)
        if move_obj not in game_board.legal_moves:
            print(f"ERROR: Illegal move {move_uci} by {player_name} from FEN {game_board.fen()}. Ending game.")
            return 1 if player_name.startswith("Stockfish") else -1
        
        current_pgn_node = current_pgn_node.add_variation(move_obj)
        game_board.push(move_obj)

    final_result_str = game_board.result(claim_draw=True)
    pgn_game_handler.headers["Result"] = final_result_str
    print(f"Game Over. Result: {final_result_str}")

    if final_result_str == "1-0": return 1 if our_agent_color_is_white else -1
    elif final_result_str == "0-1": return -1 if our_agent_color_is_white else 1
    return 0

# <<< NEW: `run_evaluation` now accepts the checkpoint filename as an argument >>>
def run_evaluation(checkpoint_filename: str):
    """Main function to run the full evaluation process."""
    eval_run_name = f"eval_{Path(checkpoint_filename).stem}_vs_sf_depth{STOCKFISH_GO_COMMAND_DEPTH}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pgn_save_dir = DATA_DIR / "evaluation_runs" / eval_run_name

    print(f"Starting evaluation run: {eval_run_name}")
    print(f"Device: {DEVICE}")
    pgn_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving PGNs to: {pgn_save_dir}")

    stockfish_player = None
    try:
        # <<< MODIFIED: Use the function argument instead of the global variable >>>
        model_checkpoint_path = CHECKPOINTS_DIR / checkpoint_filename
        agent_model, loaded_config = load_agent_from_checkpoint(model_checkpoint_path)
        
        stockfish_path = loaded_config.get("stockfish_path", "/usr/games/stockfish")
        print(f"Using Stockfish path from loaded config: {stockfish_path}")

        mcts_player = MCTS(network=agent_model, device=DEVICE)
        stockfish_player = initialize_stockfish_player(stockfish_path)
        
        board = chess.Board()
        results = {"wins": 0, "losses": 0, "draws": 0}
        
        for i in range(NUM_EVALUATION_GAMES):
            game_idx = i + 1
            agent_is_white = (i % 2 == 0)
            
            print(f"\n--- Starting Game {game_idx}/{NUM_EVALUATION_GAMES} (Agent as {'White' if agent_is_white else 'Black'}) ---")
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = f"Evaluation: {eval_run_name}"
            
            outcome = play_evaluation_game(agent_is_white, mcts_player, stockfish_player, board, pgn_game, stockfish_path)
            
            if outcome == 1: results["wins"] += 1
            elif outcome == -1: results["losses"] += 1
            else: results["draws"] += 1
            
            pgn_filename = pgn_save_dir / f"game_{game_idx:03d}_{'agent_white' if agent_is_white else 'agent_black'}.pgn"
            with open(pgn_filename, "w", encoding="utf-8") as f:
                f.write(str(pgn_game))
            print(f"PGN saved to {pgn_filename}")

        print("\n\n--- Evaluation Summary ---")
        print(f"Total Games Played: {NUM_EVALUATION_GAMES}")
        print(f"Agent Wins:   {results['wins']} ({results['wins']/NUM_EVALUATION_GAMES:.1%})")
        print(f"Agent Losses: {results['losses']} ({results['losses']/NUM_EVALUATION_GAMES:.1%})")
        print(f"Draws:        {results['draws']} ({results['draws']/NUM_EVALUATION_GAMES:.1%})")
    
    except Exception as e:
        print(f"\nFATAL ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if stockfish_player:
            stockfish_player.close()
        print("Evaluation script finished.")

# <<< NEW: Main execution block to parse command-line arguments >>>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained chess agent against Stockfish.")
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        required=True,
        help="Filename of the agent's checkpoint file (must be in the checkpoints directory)."
    )
    args = parser.parse_args()
    
    run_evaluation(checkpoint_filename=args.checkpoint)