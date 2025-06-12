#
# File: visualization/visualize_attention.py
#
"""
An interactive script to visualize the cross-attention weights of a ChessNetwork model.

This tool loads a model from a checkpoint and a PGN file, allowing a user to
step through the game and visualize attention patterns.
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

def draw_board(screen, board, piece_images, total_attention_per_square):
    """
    Draws the basic board, heatmap, and pieces.
    """
    if total_attention_per_square is not None and np.max(total_attention_per_square) > 0:
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

def draw_attention_lines(screen, attention_weights, piece_labels, from_square_idx, top_k):
    """
    Draws the top K attention lines from a selected piece to all other squares.
    """
    if attention_weights is None or piece_labels is None:
        return

    try:
        piece_tensor_idx = piece_labels.index(from_square_idx)
    except (ValueError, AttributeError):
        return

    piece_attention = attention_weights[piece_tensor_idx]
    
    k = min(top_k, len(piece_attention))
    if k == 0:
        return

    top_k_values, top_k_indices = torch.topk(piece_attention, k)
    top_k_values = top_k_values.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()

    if np.max(top_k_values) > 0:
        normalized_weights = top_k_values / np.max(top_k_values)
    else:
        return

    from_row, from_col = divmod(from_square_idx, 8)
    from_pos = ((from_col + 0.5) * SQUARE_SIZE, (7 - from_row + 0.5) * SQUARE_SIZE)

    for i in range(k):
        to_square_idx = top_k_indices[i]
        weight = normalized_weights[i]
        
        to_row, to_col = divmod(to_square_idx, 8)
        to_pos = ((to_col + 0.5) * SQUARE_SIZE, (7 - to_row + 0.5) * SQUARE_SIZE)

        line_width = int(1 + weight * 8)
        pygame.draw.line(screen, COLOR_ATTENTION_LINE, from_pos, to_pos, line_width)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively visualize cross-attention weights from a ChessNetwork model using a PGN file."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint (.pth.tar file)."
    )
    parser.add_argument(
        "--pgn_path",
        type=str,
        required=True,
        help="Path to the PGN file to visualize."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Display the top K attention lines for a selected piece."
    )
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
    
    # --- This block now gets updated in the loop ---
    attention_weights, piece_labels, total_attention_per_square = None, None, None
    
    def update_attention_data():
        nonlocal attention_weights, piece_labels, total_attention_per_square
        attention_weights, piece_labels = get_attention_for_board(board, network, device)
        if attention_weights is not None:
            total_attention_per_square = attention_weights.sum(dim=0).cpu().numpy()
        else:
            total_attention_per_square = None

    update_attention_data() # Initial call for starting position

    pygame.init()
    pygame.display.set_caption("Chess Network Attention Visualization")
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    piece_images = load_piece_images()

    selected_square_idx = None
    running = True

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
                    print(f"Moved to board state after: {board.peek() if board.move_stack else 'Start'}")
                    selected_square_idx = None # Deselect piece on move
                    update_attention_data() # Recalculate attention for new position

            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_square = pixel_to_square(event.pos)

                if clicked_square == selected_square_idx or not board.piece_at(clicked_square):
                    selected_square_idx = None
                else:
                    selected_square_idx = clicked_square

        # --- Drawing ---
        draw_board(screen, board, piece_images, total_attention_per_square)

        if selected_square_idx is not None:
            draw_attention_lines(screen, attention_weights, piece_labels, selected_square_idx, args.top_k)

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()