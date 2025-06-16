#
# File: visualization/visualize_attention.py
#
"""
An interactive script to visualize the cross-attention weights of a ChessNetwork model.

This tool loads a model from a checkpoint and a PGN file, allowing a user to
step through the game and visualize attention patterns using gradient-colored
overlays and highlights for clarity. It automatically highlights the piece
that will make the next move in the game sequence.
"""
import sys
import os

# Add the project root to the Python path to allow for package imports
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
from gnn_agent.gamestate_converters import gnn_data_converter

# --- Pygame and Board Configuration ---
SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE
ASSET_PATH = "assets/pieces"

# Colors
COLOR_LIGHT_SQ = pygame.Color(240, 217, 181)
COLOR_DARK_SQ = pygame.Color(181, 136, 99)
COLOR_SELECTED_PIECE_HIGHLIGHT = pygame.Color(255, 255, 0) # Yellow circle for the selected piece
COLOR_ATTENTION_1 = pygame.Color(255, 69, 0)      # Red-Orange for top attention
COLOR_ATTENTION_2 = pygame.Color(255, 165, 0)    # Orange for second
COLOR_ATTENTION_3 = pygame.Color(255, 215, 0)    # Gold/Yellow for third


# --- Model Configuration ---
GNN_INPUT_FEATURES = 12
GNN_HIDDEN_FEATURES = 256
GNN_OUTPUT_FEATURES = 128
NUM_ATTENTION_HEADS = 4
POLICY_HEAD_MOVE_CANDIDATES = 4672

def load_model_from_checkpoint(model_path, device):
    """
    Loads a ChessNetwork model and its weights from a .pth.tar checkpoint file.
    """
    square_gnn = SquareGNN(GNN_INPUT_FEATURES, GNN_HIDDEN_FEATURES, GNN_OUTPUT_FEATURES)
    piece_gnn = PieceGNN(GNN_INPUT_FEATURES, GNN_HIDDEN_FEATURES, GNN_OUTPUT_FEATURES)
    attention_module = CrossAttentionModule(
        sq_embed_dim=GNN_OUTPUT_FEATURES, pc_embed_dim=GNN_OUTPUT_FEATURES, num_heads=NUM_ATTENTION_HEADS
    )
    policy_head = PolicyHead(GNN_OUTPUT_FEATURES, POLICY_HEAD_MOVE_CANDIDATES)
    value_head = ValueHead(GNN_OUTPUT_FEATURES)

    network = ChessNetwork(
        square_gnn, piece_gnn, attention_module, policy_head, value_head
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    network.load_state_dict(state_dict)
    network.eval()
    print(f"Successfully loaded model from {model_path}")
    return network

def get_attention_for_board(board, network, device):
    """
    Takes a board object, processes it, and returns attention weights and piece labels.
    """
    gnn_input_data, piece_labels = gnn_data_converter.convert_to_gnn_input(
        board, device=device, for_visualization=True
    )
    with torch.no_grad():
        _, _, attention_weights = network(*gnn_input_data, return_attention=True)
    return attention_weights, piece_labels

def load_piece_images():
    """
    Loads all piece images from the asset folder into a dictionary.
    """
    images = {}
    for piece_char in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']:
        color = 'w' if piece_char.isupper() else 'b'
        filename = f"{color}{piece_char.upper()}.png"
        path = os.path.join(ASSET_PATH, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Piece image not found at {path}. Check assets folder.")
        img = pygame.image.load(path)
        images[piece_char] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
    return images

def pixel_to_square(pos):
    """Converts pixel coordinates (x, y) to a chess square index (0-63)."""
    x, y = pos
    col = x // SQUARE_SIZE
    row = 7 - (y // SQUARE_SIZE)
    return chess.square(col, row)

def draw_board_state(screen, board, piece_images, attention_weights, piece_labels, selected_square_idx, top_k):
    """
    Draws the entire board state for a frame using a robust rendering method.
    """
    # 1. Draw the board background.
    for i in range(64):
        row, col = divmod(i, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        color = COLOR_LIGHT_SQ if (row + col) % 2 == 0 else COLOR_DARK_SQ
        screen.fill(color, rect)

    # 2. If a piece is selected, draw its attention highlights.
    if selected_square_idx is not None and attention_weights is not None and piece_labels is not None:
        try:
            piece_tensor_idx = piece_labels.index(selected_square_idx)
            piece_attention = attention_weights[piece_tensor_idx]
            
            k = min(top_k, len(piece_attention))
            if k > 0:
                gradient_colors = [COLOR_ATTENTION_1, COLOR_ATTENTION_2, COLOR_ATTENTION_3]
                top_k_values, top_k_indices = torch.topk(piece_attention, k)
                max_score = top_k_values[0].item() if len(top_k_values) > 0 else 0

                for i in range(len(top_k_indices)):
                    to_square_idx = top_k_indices[i].item()
                    score = top_k_values[i].item()
                    color = gradient_colors[i] if i < len(gradient_colors) else gradient_colors[-1]
                    alpha = int(80 + (score / max_score) * 150) if max_score > 0 else 80
                    
                    to_row, to_col = divmod(to_square_idx, 8)
                    to_rect = pygame.Rect(to_col * SQUARE_SIZE, (7 - to_row) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                    
                    overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    overlay.fill((*color[:3], alpha))
                    screen.blit(overlay, to_rect.topleft)

                from_row, from_col = divmod(selected_square_idx, 8)
                from_rect = pygame.Rect(from_col * SQUARE_SIZE, (7 - from_row) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.circle(screen, COLOR_SELECTED_PIECE_HIGHLIGHT, from_rect.center, SQUARE_SIZE // 2 - 2, 4)
        except (ValueError, AttributeError):
            pass

    # 3. Draw all the pieces on top.
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            py_row = 7 - row
            rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            piece_img = piece_images[piece.symbol()]
            img_rect = piece_img.get_rect(center=rect.center)
            screen.blit(piece_img, img_rect)

def main():
    parser = argparse.ArgumentParser(
        description="Interactively visualize cross-attention weights from a ChessNetwork model."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pgn_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        with open(args.pgn_path) as pgn:
            game = chess.pgn.read_game(pgn)
    except FileNotFoundError:
        print(f"Error: PGN file not found at {args.pgn_path}")
        return
    if game is None:
        print("Error: Could not read a valid game from the PGN file.")
        return

    moves = list(game.mainline_moves())
    board = game.board() 
    move_index = -1 

    network = load_model_from_checkpoint(args.model_path, device)
    
    attention_weights, piece_labels = None, None
    
    def update_attention_data():
        nonlocal attention_weights, piece_labels
        attention_weights, piece_labels = get_attention_for_board(board, network, device)

    update_attention_data()

    pygame.init()
    pygame.display.set_caption("Chess Network Attention Visualization")
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    piece_images = load_piece_images()
    selected_square_idx = None
    
    def get_default_highlight_square():
        if -1 <= move_index < len(moves) - 1:
            return moves[move_index + 1].from_square
        return None

    selected_square_idx = get_default_highlight_square()

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                board_changed = False
                if event.key == pygame.K_RIGHT:
                    if move_index < len(moves) - 1:
                        move_index += 1
                        board.push(moves[move_index])
                        board_changed = True
                elif event.key == pygame.K_LEFT:
                    if move_index > -1:
                        board.pop()
                        move_index -= 1
                        board_changed = True
                if board_changed:
                    update_attention_data()
                    selected_square_idx = get_default_highlight_square()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_square = pixel_to_square(event.pos)
                if selected_square_idx == clicked_square:
                    selected_square_idx = get_default_highlight_square()
                elif board.piece_at(clicked_square):
                    selected_square_idx = clicked_square
                else:
                    selected_square_idx = get_default_highlight_square()

        draw_board_state(
            screen, board, piece_images, attention_weights,
            piece_labels, selected_square_idx, args.top_k
        )
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()
