#
# File: visualization/export_game_analysis.py
#
"""
A non-interactive script to generate analysis artifacts for a completed game.

This tool now only generates the individual PNG frames for a game analysis
and provides the ImageMagick command to assemble them into a GIF.
It creates a 2-frame sequence for each move (thinking + action) for clarity.
Attention is visualized with a color gradient (Red->Orange->Yellow).
"""
import sys
import os
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import chess
import chess.pgn
import pygame
import numpy as np

from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters import gnn_data_converter, action_space_converter

# --- Pygame and Board Configuration ---
SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE
INFO_PANE_HEIGHT = 80
TOTAL_HEIGHT = BOARD_SIZE + INFO_PANE_HEIGHT
ASSET_PATH = "assets/pieces"

# Colors
COLOR_LIGHT_SQ = pygame.Color(240, 217, 181)
COLOR_DARK_SQ = pygame.Color(181, 136, 99)
COLOR_INFO_BG = pygame.Color(20, 20, 20)
COLOR_INFO_FONT = pygame.Color(230, 230, 230)
COLOR_ATTENTION_FONT = pygame.Color(180, 210, 255)
COLOR_MOVING_PIECE_HIGHLIGHT = pygame.Color(255, 255, 0, 150) # Yellow for the piece that is "thinking"

# NEW: Gradient colors for attention priority
COLOR_ATTENTION_1 = pygame.Color(255, 0, 0)      # Red for top attention
COLOR_ATTENTION_2 = pygame.Color(255, 165, 0)    # Orange for second
COLOR_ATTENTION_3 = pygame.Color(255, 255, 0)    # Yellow for third


def load_model_from_checkpoint(model_path, device):
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
    print(f"Successfully loaded model from {model_path}")
    return network


def get_model_outputs_for_board(board, network, device):
    gnn_input_data = gnn_data_converter.convert_to_gnn_input(board, device=device)
    with torch.no_grad():
        policy_logits, value_logit, attention_weights = network(*gnn_input_data, return_attention=True)
    value = torch.tanh(value_logit).item()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return [], value, attention_weights, gnn_input_data
    legal_move_indices = [action_space_converter.move_to_index(m, board.turn) for m in legal_moves]
    policy_probs = torch.softmax(policy_logits.flatten(), dim=0)
    legal_probs = policy_probs[legal_move_indices]
    k = min(5, len(legal_moves))
    top_k_probs, top_k_indices_in_legal = torch.topk(legal_probs, k)
    top_moves = [(board.san(legal_moves[i]), p.item()) for i, p in zip(top_k_indices_in_legal, top_k_probs)]
    return top_moves, value, attention_weights, gnn_input_data


def format_attention_string(piece_attention_vector, from_square_san, k=3):
    if piece_attention_vector is None:
        return "Attention: Not available"
    top_k_scores, top_k_indices = torch.topk(piece_attention_vector, k)
    attended_sq_info = [f"{chess.square_name(idx.item())} ({s.item():.3f})" for s, idx in zip(top_k_scores, top_k_indices)]
    return f"Attention Focus ({from_square_san}): Top {k} squares: {', '.join(attended_sq_info)}"


def load_piece_images():
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


def draw_frame(surface, board, piece_images, move_info, attention_info, piece_attention_vector=None, moving_piece_square=None, k=3):
    """
    Draws a single frame for the GIF, including board, info pane, and visual attention.
    """
    # Draw board and highlights
    for i in range(64):
        row, col = divmod(i, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        color = COLOR_LIGHT_SQ if (row + col) % 2 == 0 else COLOR_DARK_SQ
        surface.fill(color, rect)

    # Draw visual attention overlay for attended squares
    if piece_attention_vector is not None:
        gradient_colors = [COLOR_ATTENTION_1, COLOR_ATTENTION_2, COLOR_ATTENTION_3]
        top_k_scores, top_k_indices = torch.topk(piece_attention_vector, k)
        max_score = top_k_scores[0].item() if len(top_k_scores) > 0 else 0
        
        for i in range(len(top_k_indices)):
            sq_idx = top_k_indices[i].item()
            score = top_k_scores[i].item()
            
            # MODIFIED: Select color from the gradient based on rank (i)
            color = gradient_colors[i]
            
            # Normalize alpha based on the max of the top-k scores for better visual contrast
            alpha = int(100 + (score / max_score) * 155) if max_score > 0 else 100
            
            row, col = divmod(sq_idx, 8)
            py_row = 7 - row
            rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            overlay.fill((*color[:3], alpha))
            surface.blit(overlay, rect.topleft)

    # Highlight the square of the piece that is about to move
    if moving_piece_square is not None:
        row, col = divmod(moving_piece_square, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.circle(surface, COLOR_MOVING_PIECE_HIGHLIGHT, rect.center, SQUARE_SIZE // 2, 5)

    # Draw pieces on top of board and highlights
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            py_row = 7 - row
            rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            piece_img = piece_images[piece.symbol()]
            img_rect = piece_img.get_rect(center=rect.center)
            surface.blit(piece_img, img_rect)

    # Draw info pane
    info_rect = pygame.Rect(0, BOARD_SIZE, BOARD_SIZE, INFO_PANE_HEIGHT)
    surface.fill(COLOR_INFO_BG, info_rect)
    font_main = pygame.font.SysFont('monospace', 18)
    font_attn = pygame.font.SysFont('monospace', 16)
    text_surface_main = font_main.render(move_info, True, COLOR_INFO_FONT)
    surface.blit(text_surface_main, (10, BOARD_SIZE + 10))
    text_surface_attn = font_attn.render(attention_info, True, COLOR_ATTENTION_FONT)
    surface.blit(text_surface_attn, (10, BOARD_SIZE + 40))


def main():
    parser = argparse.ArgumentParser(description="Generate analysis frames and log for a game.")
    parser.add_argument("--model_path", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--pgn_path", required=True, help="Path to the PGN file to analyze.")
    parser.add_argument("--output_dir", default="analysis_output", help="Directory to save artifacts.")
    parser.add_argument("--no-loop", action="store_true", help="Provide this flag for the GIF to play only once.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    pgn_filename = os.path.splitext(os.path.basename(args.pgn_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    frame_dir_name = f"{pgn_filename}_frames_{timestamp}"
    frame_dir_path = os.path.join(args.output_dir, frame_dir_name)
    os.makedirs(frame_dir_path, exist_ok=True)

    with open(args.pgn_path) as pgn_file:
        game = chess.pgn.read_game(pgn_file)
    if not game:
        print("Error: Could not read a valid game from the PGN file.")
        return

    network = load_model_from_checkpoint(args.model_path, device)
    board = game.board()
    pygame.init()
    piece_images = load_piece_images()
    surface = pygame.Surface((BOARD_SIZE, TOTAL_HEIGHT))
    
    frame_count = 0
    annotation_log = []

    ply = 0
    top_moves, value, _, _ = get_model_outputs_for_board(board, network, device)
    move_info_text = f"Ply {ply}: Start Position | Value: {value:.3f}"
    log_entry = f"--- Ply {ply} (Start) ---\nValue: {value:.3f}\nTop Policy Moves: {top_moves}\n"
    draw_frame(surface, board, piece_images, move_info_text, "N/A")
    frame_path = os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png")
    pygame.image.save(surface, frame_path)
    annotation_log.append(log_entry)

    for move in game.mainline_moves():
        ply += 1
        frame_count +=1
        san_move = board.san(move)
        
        _, _, pre_move_attention, gnn_data = get_model_outputs_for_board(board, network, device)
        
        piece_attention_vector = None
        from_square = move.from_square
        piece_pos_tensor = gnn_data.piece_to_square_map
        moved_piece_indices = (piece_pos_tensor == from_square).nonzero(as_tuple=True)[0]
        if moved_piece_indices.numel() > 0:
            piece_idx_in_tensor = moved_piece_indices[0]
            piece_attention_vector = pre_move_attention[piece_idx_in_tensor]

        attention_log_string = format_attention_string(piece_attention_vector, chess.square_name(from_square))

        # --- FRAME 1: "Thinking" Frame ---
        thinking_info_text = f"Ply {ply}: Thinking about {san_move}..."
        draw_frame(surface, board, piece_images, thinking_info_text, attention_log_string, piece_attention_vector, moving_piece_square=from_square)
        frame_path = os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png")
        pygame.image.save(surface, frame_path)

        board.push(move)
        frame_count += 1

        # --- FRAME 2: "Action" Frame ---
        top_moves, value, _, _ = get_model_outputs_for_board(board, network, device)
        action_info_text = f"Ply {ply}: {san_move} | Value: {value:.3f}"
        draw_frame(surface, board, piece_images, action_info_text, "")
        frame_path = os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png")
        pygame.image.save(surface, frame_path)
        
        log_entry = f"--- Ply {ply} ({san_move}) ---\nValue: {value:.3f}\nTop Policy Moves: {top_moves}\n{attention_log_string}\n"
        annotation_log.append(log_entry)
        print(f"Processed ply {ply}: {san_move}")

    pygame.quit()

    print("\nProcessing complete. Generating log file...")
    log_path = os.path.join(args.output_dir, f"{frame_dir_name}_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Analysis Log for {pgn_filename}.pgn\nModel: {args.model_path}\n{'='*40}\n\n" + "\n".join(annotation_log))
    print(f"-> Saved Annotation Log to: {log_path}")
    print("\n---")

    gif_path = os.path.join(args.output_dir, f"{frame_dir_name}.gif")
    loop_option = 1 if args.no_loop else 0
    delay = 150 # 1.5 seconds per frame
    print("✅ PNG frames and log file have been generated successfully.")
    print("To create the final GIF, please run the following command in your terminal:")
    print("\n---")
    print(f"convert -delay {delay} -loop {loop_option} '{frame_dir_path}/frame_*.png' '{gif_path}'")
    print("---\n")

if __name__ == '__main__':
    main()