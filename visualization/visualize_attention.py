#
# File: visualization/visualize_attention.py
#
"""
An interactive script to visualize the symmetric cross-attention weights of a ChessNetwork model.

This tool loads a model from a checkpoint and a PGN file, allowing a user to
step through the game and visualize attention patterns. Press 'T' to toggle
between Piece->Square and Square->Piece attention views.
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
HEADER_HEIGHT = 40 # Space for text display
ASSET_PATH = "assets/pieces"

# Colors
COLOR_LIGHT_SQ = pygame.Color(240, 217, 181)
COLOR_DARK_SQ = pygame.Color(181, 136, 99)
COLOR_HEADER_BG = pygame.Color(40, 40, 40)
COLOR_HEADER_TEXT = pygame.Color(255, 255, 255)
COLOR_SELECTED_PIECE_HIGHLIGHT = pygame.Color(255, 255, 0)
COLOR_ATTENTION_1 = pygame.Color(255, 69, 0)      # Red-Orange for top attention
COLOR_ATTENTION_2 = pygame.Color(255, 165, 0)     # Orange for second
COLOR_ATTENTION_3 = pygame.Color(255, 215, 0)     # Gold/Yellow for third


# --- Model Configuration ---
SQUARE_GNN_IN_FEATURES = 12
SQUARE_GNN_HIDDEN_FEATURES = 256
SQUARE_GNN_OUT_FEATURES = 128
SQUARE_GNN_HEADS = 4

PIECE_GNN_IN_CHANNELS = 12
PIECE_GNN_HIDDEN_CHANNELS = 256
PIECE_GNN_OUT_CHANNELS = 128

CROSS_ATTENTION_NUM_HEADS = 4
CROSS_ATTENTION_DROPOUT = 0.1

POLICY_HEAD_EMBEDDING_DIM = 128
POLICY_HEAD_MOVE_CANDIDATES = 4672
VALUE_HEAD_EMBEDDING_DIM = 128


def load_model_from_checkpoint(model_path, device):
    """
    Loads a ChessNetwork model and its weights from a .pth.tar checkpoint file.
    """
    square_gnn = SquareGNN(
        in_features=SQUARE_GNN_IN_FEATURES,
        hidden_features=SQUARE_GNN_HIDDEN_FEATURES,
        out_features=SQUARE_GNN_OUT_FEATURES,
        heads=SQUARE_GNN_HEADS
    )
    piece_gnn = PieceGNN(
        in_channels=PIECE_GNN_IN_CHANNELS,
        hidden_channels=PIECE_GNN_HIDDEN_CHANNELS,
        out_channels=PIECE_GNN_OUT_CHANNELS
    )
    attention_module = CrossAttentionModule(
        sq_embed_dim=SQUARE_GNN_OUT_FEATURES,
        pc_embed_dim=PIECE_GNN_OUT_CHANNELS,
        num_heads=CROSS_ATTENTION_NUM_HEADS,
        dropout_rate=CROSS_ATTENTION_DROPOUT
    )
    policy_head = PolicyHead(POLICY_HEAD_EMBEDDING_DIM, POLICY_HEAD_MOVE_CANDIDATES)
    value_head = ValueHead(VALUE_HEAD_EMBEDDING_DIM)

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
    Takes a board object, processes it, and returns both sets of attention weights.
    """
    gnn_input_data, piece_labels = gnn_data_converter.convert_to_gnn_input(
        board, device=device, for_visualization=True
    )
    with torch.no_grad():
        _, _, ps_weights, sp_weights = network(*gnn_input_data, return_attention=True)

    return ps_weights, sp_weights, piece_labels

def load_piece_images():
    """ Loads all piece images from the asset folder into a dictionary. """
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
    """ Converts pixel coordinates (x, y) to a chess square index (0-63). """
    x, y = pos
    if y < HEADER_HEIGHT:
        return None
    col = x // SQUARE_SIZE
    row = 7 - ((y - HEADER_HEIGHT) // SQUARE_SIZE)
    return chess.square(col, row)

def draw_board_state(screen, board, piece_images, active_attention_weights, piece_labels, selected_square_idx, top_k, view_mode):
    """ Draws the entire board state for a frame. """
    for i in range(64):
        row, col = divmod(i, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE + HEADER_HEIGHT, SQUARE_SIZE, SQUARE_SIZE)
        color = COLOR_LIGHT_SQ if (row + col) % 2 == 0 else COLOR_DARK_SQ
        screen.fill(color, rect)

    if selected_square_idx is not None and active_attention_weights is not None and piece_labels is not None:
        try:
            if view_mode == 'P->S':
                piece_tensor_idx = piece_labels.index(selected_square_idx)
                piece_attention = active_attention_weights[piece_tensor_idx]
                scores, target_indices = torch.topk(piece_attention, min(top_k, len(piece_attention)))
            elif view_mode == 'S->P':
                # --- THIS IS THE FIX ---
                # Correctly index the attention tensor for S->P mode.
                # We select the row corresponding to the selected square.
                square_attention = active_attention_weights[selected_square_idx]
                scores, target_indices_tensors = torch.topk(square_attention, min(top_k, len(square_attention)))
                # The targets are pieces, so we map their tensor indices back to square indices for drawing.
                target_indices = [piece_labels[i] for i in target_indices_tensors]

            k = min(top_k, len(scores))
            if k > 0:
                gradient_colors = [COLOR_ATTENTION_1, COLOR_ATTENTION_2, COLOR_ATTENTION_3]
                max_score = scores[0].item() if len(scores) > 0 else 0

                for i in range(k):
                    # This logic now correctly handles both a list of ints (from S->P) and a tensor (from P->S)
                    target_idx = target_indices[i] if isinstance(target_indices, list) else target_indices[i].item()
                    score = scores[i].item()
                    color = gradient_colors[i] if i < len(gradient_colors) else gradient_colors[-1]
                    alpha = int(80 + (score / max_score) * 150) if max_score > 0 else 80

                    row, col = divmod(target_idx, 8)
                    target_rect = pygame.Rect(col * SQUARE_SIZE, (7 - row) * SQUARE_SIZE + HEADER_HEIGHT, SQUARE_SIZE, SQUARE_SIZE)

                    overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    overlay.fill((*color[:3], alpha))
                    screen.blit(overlay, target_rect.topleft)

            from_row, from_col = divmod(selected_square_idx, 8)
            from_rect = pygame.Rect(from_col * SQUARE_SIZE, (7 - from_row) * SQUARE_SIZE + HEADER_HEIGHT, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.circle(screen, COLOR_SELECTED_PIECE_HIGHLIGHT, from_rect.center, SQUARE_SIZE // 2 - 2, 4)
        except (ValueError, AttributeError, IndexError):
            pass

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            py_row = 7 - row
            rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE + HEADER_HEIGHT, SQUARE_SIZE, SQUARE_SIZE)
            piece_img = piece_images[piece.symbol()]
            img_rect = piece_img.get_rect(center=rect.center)
            screen.blit(piece_img, img_rect)

def draw_header(screen, font, view_mode):
    """ Draws the header text indicating the current view mode. """
    screen.fill(COLOR_HEADER_BG, pygame.Rect(0, 0, BOARD_SIZE, HEADER_HEIGHT))
    mode_text = f"Mode: {'Piece -> Square' if view_mode == 'P->S' else 'Square -> Piece'} (Press 'T' to toggle)"
    text_surface = font.render(mode_text, True, COLOR_HEADER_TEXT)
    text_rect = text_surface.get_rect(center=(BOARD_SIZE // 2, HEADER_HEIGHT // 2))
    screen.blit(text_surface, text_rect)

def main():
    parser = argparse.ArgumentParser(description="Interactively visualize symmetric cross-attention weights.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pgn_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        with open(args.pgn_path) as pgn: game = chess.pgn.read_game(pgn)
    except FileNotFoundError:
        print(f"Error: PGN file not found at {args.pgn_path}"); return
    if game is None:
        print("Error: Could not read a valid game from the PGN file."); return

    moves = list(game.mainline_moves())
    board = game.board()
    move_index = -1

    network = load_model_from_checkpoint(args.model_path, device)

    ps_attn_weights, sp_attn_weights, piece_labels = None, None, None
    view_mode = 'P->S'

    def update_attention_data():
        nonlocal ps_attn_weights, sp_attn_weights, piece_labels
        ps_attn_weights, sp_attn_weights, piece_labels = get_attention_for_board(board, network, device)

    update_attention_data()

    pygame.init()
    pygame.display.set_caption("Symmetric Chess Attention Visualization")
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE + HEADER_HEIGHT))
    font = pygame.font.Font(None, 24)
    piece_images = load_piece_images()

    def get_default_highlight_square():
        if -1 <= move_index < len(moves) - 1:
            return moves[move_index + 1].from_square
        return None

    selected_square_idx = get_default_highlight_square()

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                board_changed = False
                if event.key == pygame.K_RIGHT:
                    if move_index < len(moves) - 1:
                        move_index += 1; board.push(moves[move_index]); board_changed = True
                elif event.key == pygame.K_LEFT:
                    if move_index > -1:
                        board.pop(); move_index -= 1; board_changed = True
                elif event.key == pygame.K_t:
                    view_mode = 'S->P' if view_mode == 'P->S' else 'P->S'
                    default_square = get_default_highlight_square()
                    if default_square is not None:
                        selected_square_idx = default_square

                if board_changed:
                    update_attention_data()
                    selected_square_idx = get_default_highlight_square()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_square = pixel_to_square(event.pos)
                if clicked_square is not None:
                    is_piece = board.piece_at(clicked_square) is not None
                    if view_mode == 'P->S' and not is_piece:
                        selected_square_idx = get_default_highlight_square()
                    elif clicked_square == selected_square_idx:
                        selected_square_idx = get_default_highlight_square()
                    else:
                        selected_square_idx = clicked_square

        active_weights = ps_attn_weights if view_mode == 'P->S' else sp_attn_weights

        screen.fill(COLOR_HEADER_BG)
        draw_header(screen, font, view_mode)
        draw_board_state(screen, board, piece_images, active_weights, piece_labels, selected_square_idx, args.top_k, view_mode)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()