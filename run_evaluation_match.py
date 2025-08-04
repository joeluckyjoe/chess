# FILENAME: run_evaluation_match.py

import torch
import chess
import chess.pgn
import argparse
import sys
from pathlib import Path
import datetime
from stockfish import Stockfish
from collections import deque
from tqdm import tqdm

# --- Project-specific Imports ---
from config import get_paths, config_params
from hardware_setup import get_device
from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.neural_network.temporal_model import TemporalPolicyValueModel
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input

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
SEQUENCE_LENGTH = 8

def load_player(player_string: str, device: torch.device):
    if player_string.lower() == 'stockfish':
        print("Loading player: Stockfish engine...")
        try:
            stockfish_path = config_params.get('STOCKFISH_PATH', '/usr/games/stockfish')
            player = Stockfish(path=stockfish_path, depth=config_params.get('STOCKFISH_DEPTH_EVAL', 10))
            return player, "Stockfish"
        except Exception as e:
            print(f"[FATAL] Could not initialize Stockfish engine: {e}")
            sys.exit(1)
            
    elif player_string.endswith(('.pth', '.pth.tar')):
        checkpoint_path = Path(player_string)
        if not checkpoint_path.exists():
            paths = get_paths()
            checkpoint_path = paths.checkpoints_dir / player_string
            if not checkpoint_path.exists():
                print(f"[FATAL] Checkpoint file not found: {player_string}")
                sys.exit(1)

        print(f"Loading player: Model from checkpoint '{checkpoint_path.name}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        is_temporal = any(key.startswith('transformer_encoder') for key in model_state_dict.keys())

        if is_temporal:
            print("Detected Temporal Model.")
            encoder_model = PolicyValueModel(
                gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
                embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
                gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
            )
            model = TemporalPolicyValueModel(
                encoder_model=encoder_model,
                policy_size=get_action_space_size(),
                d_model=config_params['EMBED_DIM']
            ).to(device)
        else:
            print("Detected GNN Model.")
            model = PolicyValueModel(
                gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
                embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
                gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
            ).to(device)
        
        model.load_state_dict(model_state_dict)
        model.eval()

        mcts_player = MCTS(
            network=model, device=device,
            c_puct=config_params['CPUCT'],
            batch_size=config_params['BATCH_SIZE']
        )
        return mcts_player, f"Agent ({checkpoint_path.stem})"
    else:
        print(f"[FATAL] Invalid player specified: '{player_string}'.")
        sys.exit(1)

def get_move(player, board: chess.Board, num_simulations: int, device: torch.device) -> chess.Move:
    if isinstance(player, Stockfish):
        player.set_fen_position(board.fen())
        best_move_uci = player.get_best_move()
        return chess.Move.from_uci(best_move_uci) if best_move_uci else None
        
    elif isinstance(player, MCTS):
        if player.is_temporal:
            initial_gnn, initial_cnn, _ = convert_to_gnn_input(chess.Board(), device)
            state_sequence_queue = deque([(initial_gnn, initial_cnn)] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
            policy, _ = player.run_search(board, num_simulations, state_sequence=list(state_sequence_queue))
        else:
            policy, _ = player.run_search(board, num_simulations)
            
        return player.select_move(policy, temperature=0.0)
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Run an evaluation match between two players.")
    parser.add_argument('--white', type=str, required=True, help="White player: 'stockfish' or path to a model checkpoint.")
    parser.add_argument('--black', type=str, required=True, help="Black player: 'stockfish' or path to a model checkpoint.")
    parser.add_argument('--games', type=int, default=10, help="Number of games to play.")
    parser.add_argument('--sims', type=int, default=config_params.get('MCTS_SIMULATIONS', 400), help="Number of MCTS simulations for agent players.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    player_white, white_name = load_player(args.white, device)
    player_black, black_name = load_player(args.black, device)
    
    scores = {"White": 0, "Black": 0, "Draw": 0}

    print("\n--- Starting Evaluation Match ---")
    print(f"White: {white_name}")
    print(f"Black: {black_name}")
    print(f"Games: {args.games}")
    print("-" * 35)

    for game_num in tqdm(range(1, args.games + 1), desc="Playing Games"):
        board = chess.Board()
        while not board.is_game_over(claim_draw=True):
            current_player = player_white if board.turn == chess.WHITE else player_black
            move = get_move(current_player, board, args.sims, device)
            if move is None: break
            board.push(move)

        result = board.result(claim_draw=True)
        if result == "1-0":
            scores["White"] += 1
        elif result == "0-1":
            scores["Black"] += 1
        else:
            scores["Draw"] += 1

    print("\n" + "-" * 35)
    print("--- Final Score ---")
    print(f"{white_name} (White): {scores['White']}")
    print(f"{black_name} (Black): {scores['Black']}")
    print(f"Draws: {scores['Draw']}")
    print("-" * 35)

if __name__ == "__main__":
    main()