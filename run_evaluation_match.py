# FILENAME: run_evaluation_match.py

import torch
import chess
import chess.pgn
import argparse
import sys
from pathlib import Path
from stockfish import Stockfish
import datetime

# --- Project-specific Imports ---
# Assuming a standard project structure where this script is in the root
from config import get_paths, config_params
from hardware_setup import get_device
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size

# --- GNN Metadata (copied for self-containment) ---
GNN_METADATA = (
    ['square', 'piece'],
    [
        ('square', 'adjacent_to', 'square'),
        ('piece', 'occupies', 'square'),
        ('piece', 'attacks', 'piece'),
        ('piece', 'defends', 'piece')
    ]
)

def load_player(player_string: str, device: torch.device):
    """
    Loads a player based on a string identifier.
    Returns either a Stockfish instance or an MCTS instance wrapping a loaded model.
    """
    if player_string.lower() == 'stockfish':
        print(f"Loading player: Stockfish engine...")
        try:
            stockfish_path = config_params.get('STOCKFISH_PATH', '/usr/games/stockfish')
            player = Stockfish(path=stockfish_path, depth=config_params.get('STOCKFISH_DEPTH_EVAL', 10))
            return player
        except Exception as e:
            print(f"[FATAL] Could not initialize Stockfish engine: {e}")
            sys.exit(1)
            
    elif player_string.endswith(('.pth', '.pth.tar')):
        checkpoint_path = Path(player_string)
        if not checkpoint_path.exists():
            # Try to resolve relative to the checkpoints directory
            paths = get_paths()
            checkpoint_path = paths.checkpoints_dir / player_string
            if not checkpoint_path.exists():
                print(f"[FATAL] Checkpoint file not found at either path: {player_string}")
                sys.exit(1)

        print(f"Loading player: Model from checkpoint '{checkpoint_path.name}'...")
        
        # Instantiate the network architecture
        model = ValueNextStateModel(
            gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'],
            cnn_in_channels=14, 
            embed_dim=config_params['EMBED_DIM'],
            policy_size=get_action_space_size(),
            gnn_num_heads=config_params['GNN_NUM_HEADS'],
            gnn_metadata=GNN_METADATA
        ).to(device)
        
        # Load the learned weights
        # Handle both full checkpoints and simple state_dict files
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()

        # Wrap the model in an MCTS search object
        mcts_player = MCTS(
            network=model,
            device=device,
            c_puct=config_params['CPUCT'],
            batch_size=config_params['BATCH_SIZE']
        )
        return mcts_player
    else:
        print(f"[FATAL] Invalid player specified: '{player_string}'. Must be 'stockfish' or a path to a model checkpoint.")
        sys.exit(1)

def get_move(player, board: chess.Board, num_simulations: int) -> chess.Move:
    """
    Gets a move from the given player object (either Stockfish or MCTS).
    """
    if isinstance(player, Stockfish):
        player.set_fen_position(board.fen())
        best_move_uci = player.get_best_move()
        if best_move_uci:
            return chess.Move.from_uci(best_move_uci)
        return None # Should not happen unless game is over
        
    elif isinstance(player, MCTS):
        policy = player.run_search(board, num_simulations)
        # For evaluation, we want the best move, so temperature is 0
        return player.select_move(policy, temperature=0.0)
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Run an evaluation match between two players.")
    parser.add_argument('--white', type=str, required=True, help="White player: 'stockfish' or path to a model checkpoint.")
    parser.add_argument('--black', type=str, required=True, help="Black player: 'stockfish' or path to a model checkpoint.")
    parser.add_argument('--sims', type=int, default=config_params.get('MCTS_SIMULATIONS', 400), help="Number of MCTS simulations for agent players.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    player_white = load_player(args.white, device)
    player_black = load_player(args.black, device)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Evaluation Match"
    game.headers["Site"] = "Juprelle, Wallonia, Belgium"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = f"Agent ({Path(args.white).name})" if 'stockfish' not in args.white.lower() else "Stockfish"
    game.headers["Black"] = f"Agent ({Path(args.black).name})" if 'stockfish' not in args.black.lower() else "Stockfish"
    
    node = game

    print("\n--- Starting Evaluation Match ---")
    print(f"White: {game.headers['White']}")
    print(f"Black: {game.headers['Black']}")
    print("-" * 35)

    while not board.is_game_over(claim_draw=True):
        current_player = player_white if board.turn == chess.WHITE else player_black
        move = get_move(current_player, board, args.sims)

        if move is None:
            print("No move returned, ending game.")
            break
        
        san_move = board.san(move)
        print(f"{board.fullmove_number}. {'...' if board.turn == chess.BLACK else ''}{san_move}")
        node = node.add_variation(move)
        board.push(move)

    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    print("-" * 35)
    print(f"Game Over. Result: {result}")
    print("-" * 35)

    # Save the PGN
    paths = get_paths()
    white_name = Path(args.white).stem if 'stockfish' not in args.white.lower() else "stockfish"
    black_name = Path(args.black).stem if 'stockfish' not in args.black.lower() else "stockfish"
    pgn_filename = paths.pgn_games_dir / f"eval_{white_name}_vs_{black_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
    
    with open(pgn_filename, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    
    print(f"PGN saved to: {pgn_filename}")

    # Clean up Stockfish processes if they exist
    if isinstance(player_white, Stockfish):
        player_white.quit()
    if isinstance(player_black, Stockfish):
        player_black.quit()

if __name__ == "__main__":
    main()