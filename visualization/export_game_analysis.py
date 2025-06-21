#
# File: visualization/export_game_analysis.py
#
"""
A non-interactive script to generate analysis artifacts for a completed game.

This tool generates two primary outputs:
1.  Individual PNG frames for creating a visual game analysis GIF.
2.  A structured JSON Lines (.jsonl) log file containing detailed,
    machine-readable data for each move. This includes agent evaluations,
    policy, attention, and optional Stockfish evaluations.

It AUTOMATICALLY detects the latest model checkpoint and the Stockfish path
from the central config.py file.
"""
import sys
import os
from datetime import datetime
import collections
from pathlib import Path
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import chess
import chess.pgn
import chess.engine
import pygame
import numpy as np

# --- Project-specific Imports ---
from config import get_paths, config_params
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters import gnn_data_converter, action_space_converter

# --- Pygame and Board Configuration ---
SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE
INFO_PANE_HEIGHT = 120
TOTAL_HEIGHT = BOARD_SIZE + INFO_PANE_HEIGHT
ASSET_PATH = "assets/pieces"

# Colors
COLOR_LIGHT_SQ = pygame.Color(240, 217, 181)
COLOR_DARK_SQ = pygame.Color(181, 136, 99)
COLOR_INFO_BG = pygame.Color(20, 20, 20)
COLOR_INFO_FONT = pygame.Color(230, 230, 230)
COLOR_MOVING_PIECE_HIGHLIGHT = pygame.Color(255, 255, 0, 150)

# Gradient colors for attention priority
COLOR_ATTENTION_1 = pygame.Color(255, 0, 0)
COLOR_ATTENTION_2 = pygame.Color(255, 165, 0)
COLOR_ATTENTION_3 = pygame.Color(255, 255, 0)


def find_latest_checkpoint(checkpoint_dir):
    """Finds the most recently modified checkpoint file in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth.tar'))
    
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint


def load_model_from_checkpoint(model_path, device):
    """Loads a ChessNetwork model from a .pth checkpoint file."""
    square_gnn = SquareGNN(in_features=12, hidden_features=256, out_features=128, heads=4)
    piece_gnn = PieceGNN(in_channels=12, hidden_channels=256, out_channels=128)
    attention_module = CrossAttentionModule(sq_embed_dim=128, pc_embed_dim=128, num_heads=4)
    policy_head = PolicyHead(embedding_dim=128, num_possible_moves=4672)
    value_head = ValueHead(embedding_dim=128)
    network = ChessNetwork(square_gnn, piece_gnn, attention_module, policy_head, value_head).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    network.load_state_dict(state_dict)
    network.eval()
    print(f"Successfully loaded model from {os.path.basename(str(model_path))}")
    return network


def get_model_outputs_for_board(board, network, device):
    """Gets policy, value, and attention from the network for a given board state."""
    gnn_input_data = gnn_data_converter.convert_to_gnn_input(board, device=device)
    with torch.no_grad():
        policy_logits, value_logit, attention_weights = network(*gnn_input_data, return_attention=True)
    value = torch.tanh(value_logit).item()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return [], value, attention_weights, gnn_input_data

    legal_move_indices = [action_space_converter.move_to_index(m, board.turn) for m in legal_moves]
    legal_move_indices_tensor = torch.tensor(legal_move_indices, device=policy_logits.device)

    policy_probs = torch.softmax(policy_logits.flatten(), dim=0)
    legal_probs = torch.gather(policy_probs, 0, legal_move_indices_tensor)

    k = min(3, len(legal_moves))
    top_k_probs, top_k_indices_in_legal_list = torch.topk(legal_probs, k)

    top_moves = [(board.san(legal_moves[i]), p.item()) for i, p in zip(top_k_indices_in_legal_list, top_k_probs)]
    return top_moves, value, attention_weights, gnn_input_data


def format_attention_string_for_display(piece_attention_vector, from_square_san, k=3):
    """Formats the top-k attended squares into a string for Pygame display."""
    if piece_attention_vector is None:
        return "Attention: N/A"
    top_k_scores, top_k_indices = torch.topk(piece_attention_vector, k)
    attended_sq_info = [f"{chess.square_name(idx.item())}({s:.2f})" for s, idx in zip(top_k_scores, top_k_indices)]
    return f"Attn Focus ({from_square_san}): {', '.join(attended_sq_info)}"


def get_structured_attention_data(piece_attention_vector, from_square, k=3):
    """Creates a structured dictionary for the top-k attended squares for JSON logging."""
    if piece_attention_vector is None:
        return None
    top_k_scores, top_k_indices = torch.topk(piece_attention_vector, k)
    
    attended_squares = [
        {"square": chess.square_name(idx.item()), "score": round(s.item(), 5)}
        for s, idx in zip(top_k_scores, top_k_indices)
    ]
    
    return {
        "from_square": chess.square_name(from_square),
        "top_k_attended": attended_squares
    }


def load_piece_images():
    """Loads piece images from the assets directory."""
    images = {}
    for piece_char in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']:
        color = 'w' if piece_char.isupper() else 'b'
        filename = f"{color}{piece_char.upper()}.png"
        path = os.path.join(project_root, ASSET_PATH, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Piece image not found at {path}.")
        img = pygame.image.load(path)
        images[piece_char] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
    return images


def draw_frame(surface, board, piece_images, info_lines, piece_attention_vector=None, moving_piece_square=None, k=3):
    """Draws a single frame, including board, multi-line info pane, and visual attention."""
    for i in range(64):
        row, col = divmod(i, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        color = COLOR_LIGHT_SQ if (row + col) % 2 == 0 else COLOR_DARK_SQ
        surface.fill(color, rect)

    if piece_attention_vector is not None:
        gradient_colors = [COLOR_ATTENTION_1, COLOR_ATTENTION_2, COLOR_ATTENTION_3]
        top_k_scores, top_k_indices = torch.topk(piece_attention_vector, k)
        max_score = top_k_scores[0].item() if len(top_k_scores) > 0 else 0

        for i in range(len(top_k_indices)):
            sq_idx = top_k_indices[i].item()
            score = top_k_scores[i].item()
            color = gradient_colors[i]
            alpha = int(100 + (score / max_score) * 155) if max_score > 0 else 100
            row, col = divmod(sq_idx, 8)
            py_row = 7 - row
            rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            overlay.fill((*color[:3], alpha))
            surface.blit(overlay, rect.topleft)

    if moving_piece_square is not None:
        row, col = divmod(moving_piece_square, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.circle(surface, COLOR_MOVING_PIECE_HIGHLIGHT, rect.center, SQUARE_SIZE // 2, 5)

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            py_row = 7 - row
            rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            piece_img = piece_images[piece.symbol()]
            img_rect = piece_img.get_rect(center=rect.center)
            surface.blit(piece_img, img_rect)

    info_rect = pygame.Rect(0, BOARD_SIZE, BOARD_SIZE, INFO_PANE_HEIGHT)
    surface.fill(COLOR_INFO_BG, info_rect)
    font_main = pygame.font.SysFont('monospace', 16)
    y_offset = BOARD_SIZE + 5
    for line in info_lines:
        text_surface = font_main.render(line, True, COLOR_INFO_FONT)
        surface.blit(text_surface, (10, y_offset))
        y_offset += 20


def get_stockfish_eval(engine, board, time_limit=0.1):
    """Gets the centipawn evaluation from the Stockfish engine."""
    if engine is None:
        return "N/A"
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info["score"].white()
        if score.is_mate():
            return f"Mate({score.mate()})"
        else:
            return f"{score.score()}"
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as e:
        print(f"Stockfish engine error: {e}")
        return "Error"


def main():
    parser = argparse.ArgumentParser(description="Generate analysis frames and a JSONL log for a game.")
    parser.add_argument("--pgn_path", required=True, help="Path to the PGN file to analyze.")
    parser.add_argument("--model_path", type=str, default=None, help="(Optional) Path to a specific model checkpoint. If not provided, the latest checkpoint will be used.")
    parser.add_argument("--output_dir", default="analysis_output", help="Directory to save artifacts.")
    parser.add_argument("--no-loop", action="store_true", help="Provide this flag for the GIF to play only once.")
    args = parser.parse_args()

    # --- Get Paths from Config ---
    # CORRECTED: Unpack all three values from get_paths, even if not all are used.
    checkpoints_dir, _, _ = get_paths()
    stockfish_path_from_config = config_params.get("STOCKFISH_PATH")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Determine Model Path (Automatic or Manual) ---
    model_path = args.model_path
    if model_path:
        print(f"Using manually specified model: {model_path}")
    else:
        print("No model path specified. Searching for the latest checkpoint...")
        model_path = find_latest_checkpoint(checkpoints_dir)
        if not model_path:
            print(f"Error: No checkpoints found in '{checkpoints_dir}'. Please train a model first or specify a path with --model_path.")
            return
        print(f"Automatically selected latest model: {os.path.basename(str(model_path))}")

    # --- Setup Output Dirs ---
    os.makedirs(args.output_dir, exist_ok=True)
    pgn_filename = os.path.splitext(os.path.basename(args.pgn_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    frame_dir_name = f"{pgn_filename}_frames_{timestamp}"
    frame_dir_path = os.path.join(args.output_dir, frame_dir_name)
    os.makedirs(frame_dir_path, exist_ok=True)

    # --- Initialize Stockfish Engine ---
    stockfish_engine = None
    if stockfish_path_from_config and os.path.exists(stockfish_path_from_config):
        print(f"Initializing Stockfish from config path: {stockfish_path_from_config}")
        try:
            stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path_from_config)
        except Exception as e:
            print(f"Warning: Could not start Stockfish engine. Error: {e}. Continuing without it.")
    else:
        print(f"Warning: Stockfish path '{stockfish_path_from_config}' not found or not set in config. Continuing without objective evaluation.")

    # --- Main Processing Logic ---
    try:
        with open(args.pgn_path) as pgn_file:
            game = chess.pgn.read_game(pgn_file)
        if not game:
            print(f"Error: Could not read a valid game from PGN file: {args.pgn_path}")
            return

        network = load_model_from_checkpoint(model_path, device)
        board = game.board()
        pygame.init()
        piece_images = load_piece_images()
        surface = pygame.Surface((BOARD_SIZE, TOTAL_HEIGHT))
        
        frame_count = 0
        json_log_entries = []

        ply = 0
        _, pre_move_value, _, _ = get_model_outputs_for_board(board, network, device)
        stockfish_eval_str = get_stockfish_eval(stockfish_engine, board)
        
        initial_log_entry = {
            "ply": ply,
            "move_san": "Initial",
            "board_fen_before": board.fen(),
            "agent_value_before": round(pre_move_value, 5),
            "stockfish_eval": stockfish_eval_str
        }
        json_log_entries.append(initial_log_entry)
        
        info_lines = [f"Ply {ply}: Start Position", f"Agent Value: {pre_move_value:.3f}", f"Stockfish Eval: {stockfish_eval_str} cp"]
        draw_frame(surface, board, piece_images, info_lines)
        frame_path = os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png")
        pygame.image.save(surface, frame_path)
        
        for move in game.mainline_moves():
            ply += 1
            frame_count +=1
            san_move = board.san(move)
            
            pre_move_fen = board.fen()
            is_capture = board.is_capture(move)
            pre_move_policy, pre_move_value, pre_move_attention, gnn_data = get_model_outputs_for_board(board, network, device)
            stockfish_eval_str = get_stockfish_eval(stockfish_engine, board)
            
            piece_attention_vector = None
            structured_attention = None
            from_square = move.from_square
            piece_pos_tensor = gnn_data.piece_to_square_map
            moved_piece_indices = (piece_pos_tensor == from_square).nonzero(as_tuple=True)[0]
            if moved_piece_indices.numel() > 0:
                piece_idx_in_tensor = moved_piece_indices[0]
                piece_attention_vector = pre_move_attention[piece_idx_in_tensor]
                structured_attention = get_structured_attention_data(piece_attention_vector, from_square)

            policy_str_display = ', '.join([f"{m_san}({p:.2f})" for m_san, p in pre_move_policy])
            attention_str_display = format_attention_string_for_display(piece_attention_vector, chess.square_name(from_square))
            thinking_lines = [
                f"Ply {ply}: Thinking about {san_move}...",
                f"Agent Value: {pre_move_value:.3f} | Stockfish: {stockfish_eval_str} cp",
                f"Agent Policy: {policy_str_display}",
                attention_str_display
            ]
            draw_frame(surface, board, piece_images, thinking_lines, piece_attention_vector, moving_piece_square=from_square)
            frame_path = os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png")
            pygame.image.save(surface, frame_path)

            board.push(move)
            frame_count += 1
            
            is_check_after = board.is_check()
            is_mate_after = board.is_checkmate()
            _, post_move_value, _, _ = get_model_outputs_for_board(board, network, device)

            action_lines = [f"Ply {ply}: Played {san_move}", f"Resulting Agent Value: {post_move_value:.3f}", f"Value Change: {post_move_value - pre_move_value:+.3f}"]
            draw_frame(surface, board, piece_images, action_lines)
            frame_path = os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png")
            pygame.image.save(surface, frame_path)
            
            log_entry = {
                "ply": ply,
                "move_san": san_move,
                "board_fen_before": pre_move_fen,
                "move_uci": move.uci(),
                "move_characteristics": {
                    "is_capture": is_capture,
                    "is_check_after": is_check_after,
                    "is_mate_after": is_mate_after
                },
                "agent_eval": {
                    "value_before": round(pre_move_value, 5),
                    "value_after": round(post_move_value, 5),
                    "policy_before": [{"move": m, "prob": round(p, 5)} for m, p in pre_move_policy]
                },
                "stockfish_eval": stockfish_eval_str,
                "moving_piece_attention": structured_attention
            }
            json_log_entries.append(log_entry)
            print(f"Processed ply {ply}: {san_move}")

    finally:
        pygame.quit()
        if stockfish_engine:
            stockfish_engine.quit()
            print("Stockfish engine terminated.")

    print("\nProcessing complete. Generating JSON Lines log file...")
    log_path = os.path.join(args.output_dir, f"{frame_dir_name}_analysis.jsonl")
    with open(log_path, 'w') as f:
        metadata = {
            "type": "analysis_metadata",
            "pgn_file": os.path.basename(args.pgn_path),
            "model_checkpoint": os.path.basename(str(model_path)),
            "stockfish_version": stockfish_path_from_config or "N/A",
            "analysis_timestamp_utc": datetime.utcnow().isoformat()
        }
        f.write(json.dumps(metadata) + '\n')
        
        for entry in json_log_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"-> Saved Structured Analysis Log to: {log_path}")

    gif_path = os.path.join(args.output_dir, f"{pgn_filename}.gif")
    loop_option = "-loop 0" if not args.no_loop else ""
    delay = 150
    print("\n✅ PNG frames and JSONL log file have been generated successfully.")
    print("To create the final GIF, please run the following command in your terminal:")
    print(f"\n--- \nconvert -delay {delay} {loop_option} '{frame_dir_path}/frame_*.png' '{gif_path}'\n--- \n")

if __name__ == '__main__':
    main()
