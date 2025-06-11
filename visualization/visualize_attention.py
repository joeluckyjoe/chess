#
# File: visualization/visualize_attention.py
#
"""
An interactive script to visualize the cross-attention weights of a ChessNetwork model.

This tool loads a model from a checkpoint, takes a FEN string, and launches an
interactive Pygame window. Users can click on pieces to see their specific
attention patterns visualized as lines on the board.
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
import pygame
import numpy as np

from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters import gnn_data_converter

# --- Pygame and Board Configuration ---
SQUARE_SIZE = 80  # Increased for better visibility
BOARD_SIZE = 8 * SQUARE_SIZE
ASSET_PATH = "assets/pieces"

# Colors
COLOR_LIGHT_SQ = pygame.Color(240, 217, 181)
COLOR_DARK_SQ = pygame.Color(181, 136, 99)
COLOR_HEATMAP = pygame.Color(135, 206, 250, 150) # Light Sky Blue for heatmap
COLOR_ATTENTION_LINE = pygame.Color(255, 0, 0, 200) # Red for attention lines

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
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    network.load_state_dict(state_dict)
    network.eval()
    print(f"Successfully loaded model from {model_path}")
    return network

def get_attention_from_fen(board, network, device):
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

def draw_board(screen, board, piece_images, total_attention_per_square):
    """
    Draws the basic board, heatmap, and pieces.
    """
    # Draw squares and heatmap
    if np.max(total_attention_per_square) > 0:
        normalized_attention = total_attention_per_square / np.max(total_attention_per_square)
    else:
        normalized_attention = np.zeros(64)

    for i in range(64):
        row, col = divmod(i, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        base_color = COLOR_LIGHT_SQ if (row + col) % 2 == 0 else COLOR_DARK_SQ
        screen.fill(base_color, rect)

        attention_value = normalized_attention[i]
        if attention_value > 0.01:
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_color = base_color.lerp(COLOR_HEATMAP, attention_value)
            highlight_surface.fill(highlight_color)
            screen.blit(highlight_surface, rect.topleft)

    # Draw pieces
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            py_row = 7 - row
            rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            piece_img = piece_images[piece.symbol()]
            img_rect = piece_img.get_rect(center=rect.center)
            screen.blit(piece_img, img_rect)

def draw_attention_lines(screen, attention_weights, piece_labels, from_square_idx):
    """
    Draws attention lines from a selected piece to all other squares.
    """
    # Find the index in the attention tensor corresponding to our selected piece
    try:
        piece_tensor_idx = piece_labels.index(from_square_idx)
    except ValueError:
        # The selected square has no piece or is not in the label list
        return
        
    # Get the specific attention vector for this piece
    piece_attention = attention_weights[piece_tensor_idx].cpu().numpy()
    
    # Normalize for visual thickness
    if np.max(piece_attention) > 0:
        normalized_attention = piece_attention / np.max(piece_attention)
    else:
        return # No attention to draw

    from_row, from_col = divmod(from_square_idx, 8)
    from_pos = ((from_col + 0.5) * SQUARE_SIZE, (7 - from_row + 0.5) * SQUARE_SIZE)

    for to_square_idx in range(64):
        weight = normalized_attention[to_square_idx]
        if weight < 0.1: # Threshold to avoid clutter
            continue

        to_row, to_col = divmod(to_square_idx, 8)
        to_pos = ((to_col + 0.5) * SQUARE_SIZE, (7 - to_row + 0.5) * SQUARE_SIZE)
        
        # Line width based on attention weight
        line_width = int(1 + weight * 8)
        
        pygame.draw.line(screen, COLOR_ATTENTION_LINE, from_pos, to_pos, line_width)

def main():
    parser = argparse.ArgumentParser(
        description="Interactively visualize cross-attention weights from a ChessNetwork model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint (.pth.tar file)."
    )
    parser.add_argument(
        "--fen",
        type=str,
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="FEN string of the initial board position to analyze."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialization ---
    network = load_model_from_checkpoint(args.model_path, device)
    board = chess.Board(args.fen)
    attention_weights, piece_labels = get_attention_from_fen(board, network, device)

    if attention_weights is None or not piece_labels:
        print("Could not retrieve attention weights or piece labels. Exiting.")
        return

    pygame.init()
    pygame.display.set_caption("Chess Network Attention Visualization")
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    piece_images = load_piece_images()

    total_attention_per_square = attention_weights.sum(dim=0).cpu().numpy()
    
    selected_square_idx = None
    running = True

    # --- Main Game Loop ---
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_square = pixel_to_square(event.pos)
                
                # Deselect if clicking the same square or an empty square
                if clicked_square == selected_square_idx or not board.piece_at(clicked_square):
                    selected_square_idx = None
                else:
                    selected_square_idx = clicked_square

        # --- Drawing ---
        # Draw the board with the overall attention heatmap
        draw_board(screen, board, piece_images, total_attention_per_square)

        # If a piece is selected, draw its specific attention lines over the board
        if selected_square_idx is not None:
            draw_attention_lines(screen, attention_weights, piece_labels, selected_square_idx)

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()