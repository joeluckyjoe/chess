import torch
import chess
from stockfish import Stockfish
from pathlib import Path
import sys
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent))

from config import get_paths, config_params
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from hardware_setup import get_device

TARGET_POSITIONS = [
    {
        "name": "After 1. d4 Nf6",
        "moves": ["d2d4", "g8f6"],
        "player_to_move": chess.WHITE
    },
]
NUM_TOP_MOVES = 3
CORRECTIVE_DATASET_FILENAME = "corrective_dataset_v1.pt"
SEQUENCE_LENGTH = 8

def create_target_policy(board: chess.Board, top_moves: list) -> torch.Tensor:
    policy_size = get_action_space_size()
    policy_tensor = torch.zeros(policy_size, dtype=torch.float32)
    if not top_moves:
        return policy_tensor
    prob_per_move = 1.0 / len(top_moves)
    for move_dict in top_moves:
        move_uci = move_dict['Move']
        move = chess.Move.from_uci(move_uci)
        try:
            action_index = move_to_index(move, board)
            if action_index is not None:
                policy_tensor[action_index] = prob_per_move
        except Exception as e:
            print(f"Warning: Could not process move {move_uci}: {e}")
    return policy_tensor

def create_input_sequence(board: chess.Board, device: torch.device) -> list:
    temp_board = chess.Board()
    initial_gnn, initial_cnn, _ = convert_to_gnn_input(temp_board, device)
    initial_state_tuple = (initial_gnn, initial_cnn)
    state_deque = deque([initial_state_tuple] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    for move in board.move_stack:
        temp_board.push(move)
        gnn_data, cnn_tensor, _ = convert_to_gnn_input(temp_board, device)
        state_deque.append((gnn_data, cnn_tensor))
    return list(state_deque)

def main():
    print("--- Starting Corrective Dataset Generation ---")
    paths = get_paths()
    device = get_device()
    try:
        stockfish_path = config_params.get("STOCKFISH_PATH")
        if not stockfish_path or not Path(stockfish_path).exists():
            raise FileNotFoundError(f"Stockfish executable not found at '{stockfish_path}'")
        stockfish = Stockfish(path=stockfish_path, parameters={"Skill Level": 20})
        print(f"Initialized Stockfish (Skill Level: 20) from: {stockfish_path}")
    except Exception as e:
        print(f"[FATAL] Could not initialize Stockfish: {e}")
        sys.exit(1)
    corrective_data = []
    for position_info in TARGET_POSITIONS:
        print(f"\nProcessing position: '{position_info['name']}'")
        board = chess.Board()
        for move_uci in position_info['moves']:
            board.push_uci(move_uci)
        print(f"Board FEN: {board.fen()}")
        stockfish.set_fen_position(board.fen())
        top_moves = stockfish.get_top_moves(NUM_TOP_MOVES)
        if not top_moves:
            print("Warning: Stockfish did not return any top moves. Skipping position.")
            continue
        print(f"Found top {len(top_moves)} moves from Stockfish: {[m['Move'] for m in top_moves]}")
        input_sequence = create_input_sequence(board, device)
        target_policy = create_target_policy(board, top_moves)
        # <<< FIXED: Changed keys to match the batching function's expectations
        corrective_data.append({
            'state_sequence': input_sequence,
            'policy': target_policy
        })
        print("Successfully created and added data point.")
    if corrective_data:
        save_path = paths.drive_project_root / CORRECTIVE_DATASET_FILENAME
        torch.save(corrective_data, save_path)
        print(f"\n✅ Corrective dataset with {len(corrective_data)} samples saved successfully to:")
        print(save_path)
    else:
        print("\nNo data was generated. The dataset file was not created.")
    print("\n--- Generation Finished ---")

if __name__ == "__main__":
    main()