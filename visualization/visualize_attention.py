#
# File: visualization/visualize_attention.py
#
"""
A script to visualize the cross-attention weights of a trained ChessNetwork model.

This tool loads a model from a checkpoint, takes a FEN string, and generates
a graphical chessboard where squares are highlighted based on attention.
"""
import argparse
import torch
import chess
import pygame
import numpy as np
import os

from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters import gnn_data_converter

# --- Pygame and Board Configuration ---
SQUARE_SIZE = 60
BOARD_SIZE = 8 * SQUARE_SIZE
ASSET_PATH = "assets/pieces"

# Colors
COLOR_LIGHT_SQ = pygame.Color(240, 217, 181)
COLOR_DARK_SQ = pygame.Color(181, 136, 99)
COLOR_HIGHLIGHT = pygame.Color(135, 206, 250, 150) # Light Sky Blue with some transparency

# --- Model Configuration ---
GNN_INPUT_FEATURES = 12
# --- THIS IS THE FINAL FIX ---
# The GNNs use a hidden layer of 256 and output a final embedding of 128.
GNN_HIDDEN_FEATURES = 256
GNN_OUTPUT_FEATURES = 128 
NUM_ATTENTION_HEADS = 4
POLICY_HEAD_MOVE_CANDIDATES = 4672

def load_model_from_checkpoint(model_path, device):
    """
    Loads a ChessNetwork model and its weights from a .pth.tar checkpoint file.
    This function is now robust to handle different checkpoint formats and architectures.
    """
    # Use the correct, discovered dimensions to instantiate the model
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

    # Handle different checkpoint formats gracefully by finding the correct state dictionary.
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
    Takes a board object, processes it, and returns attention weights and labels.
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
            raise FileNotFoundError(f"Piece image not found at {path}. Please check your assets folder.")
        images[piece_char] = pygame.image.load(path)
    return images

def draw_attention_on_board(board, attention_weights, piece_labels, save_path):
    """
    Draws a chessboard using Pygame and overlays attention weights as highlights.
    """
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    
    total_attention_per_square = attention_weights.sum(dim=0).cpu().numpy()
    if np.max(total_attention_per_square) > 0:
        normalized_attention = total_attention_per_square / np.max(total_attention_per_square)
    else:
        normalized_attention = np.zeros(64)

    piece_images = load_piece_images()

    for i in range(64):
        row, col = divmod(i, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        base_color = COLOR_LIGHT_SQ if (row + col) % 2 == 0 else COLOR_DARK_SQ
        screen.fill(base_color, rect)

        attention_value = normalized_attention[i]
        if attention_value > 0.01:
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_color = base_color.lerp(COLOR_HIGHLIGHT, attention_value)
            highlight_surface.fill(highlight_color)
            screen.blit(highlight_surface, rect.topleft)

        piece = board.piece_at(i)
        if piece:
            piece_img = piece_images[piece.symbol()]
            img_rect = piece_img.get_rect(center=rect.center)
            screen.blit(piece_img, img_rect)
            
    pygame.image.save(screen, save_path)
    print(f"Attention board saved to {save_path}")
    pygame.quit()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize cross-attention weights from a trained ChessNetwork model."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint (.pth.tar file)."
    )
    parser.add_argument(
        "--fen", 
        type=str, 
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 
        help="FEN string of the board position to analyze."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="attention_board.png", 
        help="Path to save the output visualization PNG file."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    network = load_model_from_checkpoint(args.model_path, device)
    board = chess.Board(args.fen)
    attention_weights, piece_labels = get_attention_from_fen(board, network, device)

    if attention_weights is None or not piece_labels:
        print("Could not retrieve attention weights or piece labels. Exiting.")
        return

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    draw_attention_on_board(board, attention_weights, piece_labels, args.output_path)

if __name__ == '__main__':
    main()