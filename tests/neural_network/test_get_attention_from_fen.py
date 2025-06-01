import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Any, List
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from your project
from gnn_agent.neural_network.chess_network import ChessNetwork
# We don't import CrossAttentionModule directly here, as it's used by ChessNetwork

# --- Mock GNNs and Heads (copied from previous test script for self-containment) ---
class MockSquareGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, heads):
        super().__init__()
        self.out_features = out_features
        print(f"MockSquareGNN initialized: in={in_features}, out={out_features}")
    def forward(self, features, edge_index):
        num_squares = features.size(0)
        return torch.randn(num_squares, self.out_features)

class MockPieceGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        print(f"MockPieceGNN initialized: in={in_channels}, out={out_channels}")
    def forward(self, features, edge_index):
        num_pieces = features.size(0)
        if num_pieces == 0: return torch.empty(0, self.out_channels)
        return torch.randn(num_pieces, self.out_channels)

class MockPolicyHead(nn.Module):
    def __init__(self, embed_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        print(f"MockPolicyHead initialized: embed_dim={embed_dim}, num_actions={num_actions}")
    def forward(self, fused_embeddings_batch):
        batch_size = fused_embeddings_batch.size(0)
        return torch.randn(batch_size, self.num_actions)

class MockValueHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        print(f"MockValueHead initialized: embed_dim={embed_dim}")
    def forward(self, fused_embeddings_batch):
        batch_size = fused_embeddings_batch.size(0)
        return torch.randn(batch_size, 1)
# --- End Mock GNNs and Heads ---

class MockGNNDataConverter:
    """
    A simplified mock converter to simulate turning a FEN into GNN input tensors.
    For actual visualization, you'd use your real GNNDataConverter.
    """
    def __init__(self, square_features_dim: int, piece_features_dim: int):
        self.square_features_dim = square_features_dim
        self.piece_features_dim = piece_features_dim
        # Corrected mapping for FEN parsing (0=a8, 1=b8, ..., 63=h1)
        self.square_idx_to_algebraic = [f"{chr(ord('a') + (i % 8))}{8 - (i // 8)}" for i in range(64)] 
        self.algebraic_to_square_idx = {sq: i for i, sq in enumerate(self.square_idx_to_algebraic)}

        print("MockGNNDataConverter initialized.")

    def convert_fen_to_gnn_input(self, fen: str) -> Dict[str, Any]:
        print(f"MockGNNDataConverter: Converting FEN: {fen[:20]}...") 

        num_squares = 64
        
        piece_placement = fen.split(' ')[0]
        num_pieces = 0
        piece_labels: List[str] = []
        # For mock labels, we also store their FEN character and original square for better identification
        # List of tuples: (label_str, fen_char, algebraic_square_of_piece)
        piece_details: List[Tuple[str, str, str]] = [] 

        current_square_idx = 0 # 0=a8, 1=b8 ... 63=h1 (standard FEN parsing order)
        
        for rank_str in piece_placement.split('/'):
            file_idx_on_rank = 0 # Tracks file position within the current rank string
            for char_in_rank in rank_str:
                if char_in_rank.isalpha():
                    num_pieces += 1
                    # The `current_square_idx` correctly tracks the FEN square index (0=a8, etc.)
                    algebraic_pos = self.square_idx_to_algebraic[current_square_idx]
                    
                    label = f"{char_in_rank}_{algebraic_pos}"
                    piece_labels.append(label)
                    piece_details.append((label, char_in_rank, algebraic_pos))
                    
                    current_square_idx += 1
                    # file_idx_on_rank was not needed here as current_square_idx is global
                elif char_in_rank.isdigit():
                    skip = int(char_in_rank)
                    current_square_idx += skip
                    # file_idx_on_rank was not needed here

        print(f"MockGNNDataConverter: Estimated {num_pieces} pieces from FEN. Generated {len(piece_labels)} labels.")
        if num_pieces > 0:
            print(f"MockGNNDataConverter: First 5 piece details: {piece_details[:5]}")


        square_features = torch.randn(num_squares, self.square_features_dim)
        edges = []
        for r in range(8):
            for f in range(8):
                idx = r * 8 + f
                for dr in [-1, 0, 1]:
                    for df in [-1, 0, 1]:
                        if dr == 0 and df == 0: continue
                        nr, nf = r + dr, f + df
                        if 0 <= nr < 8 and 0 <= nf < 8:
                            n_idx = nr * 8 + nf
                            edges.append([idx, n_idx])
        square_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)

        if num_pieces > 0:
            piece_features = torch.randn(num_pieces, self.piece_features_dim)
            if num_pieces > 1:
                piece_edge_index = torch.randint(0, num_pieces, (2, num_pieces * 2), dtype=torch.long) 
            else:
                 piece_edge_index = torch.empty((2,0), dtype=torch.long)
            piece_padding_mask = torch.zeros(num_pieces, dtype=torch.bool)
        else:
            piece_features = torch.empty(0, self.piece_features_dim)
            piece_edge_index = torch.empty((2,0), dtype=torch.long)
            piece_padding_mask = torch.empty(0, dtype=torch.bool)

        piece_to_square_map_mock_tensor = torch.arange(num_pieces, dtype=torch.long) if num_pieces > 0 else torch.empty(0, dtype=torch.long)

        return {
            "square_features": square_features,
            "square_edge_index": square_edge_index,
            "piece_features": piece_features,
            "piece_edge_index": piece_edge_index,
            "piece_to_square_map": piece_to_square_map_mock_tensor, 
            "piece_padding_mask": piece_padding_mask,
            "piece_labels": piece_labels,
            "piece_details": piece_details 
        }

def get_gnn_data_and_attention(fen_string: str, 
                                network: ChessNetwork, 
                                converter: MockGNNDataConverter
                                ) -> Tuple[Optional[torch.Tensor], Optional[List[str]], Optional[List[Tuple[str,str,str]]]]:
    gnn_input_data = converter.convert_fen_to_gnn_input(fen_string)
    piece_labels = gnn_input_data.get("piece_labels")
    piece_details = gnn_input_data.get("piece_details")

    network.eval() 
    with torch.no_grad():
        sf = gnn_input_data["square_features"]
        sei = gnn_input_data["square_edge_index"]
        pf = gnn_input_data["piece_features"]
        pei = gnn_input_data["piece_edge_index"]
        ptsm = gnn_input_data["piece_to_square_map"] 
        ppm = gnn_input_data["piece_padding_mask"]

        if pf is None or pf.nelement() == 0:
             print("No piece features, expecting None for attention weights.")
        
        policy_logits, value, attention_weights = network.forward(
            square_features=sf, square_edge_index=sei,
            piece_features=pf, piece_edge_index=pei,
            piece_to_square_map=ptsm, piece_padding_mask=ppm,
            return_attention_weights=True
        )
    
    return attention_weights, piece_labels, piece_details

def plot_square_attention_to_pieces(attention_values: np.ndarray, 
                                     piece_labels: List[str], 
                                     square_label: str,
                                     fen: str):
    if not piece_labels:
        print(f"No pieces to plot attention for square {square_label}.")
        return
    if attention_values.size == 0 :
        print(f"Attention values are empty for square {square_label}.")
        return
    if len(attention_values) != len(piece_labels):
        print(f"Mismatch between attention values ({len(attention_values)}) and piece labels ({len(piece_labels)}). Cannot plot.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(piece_labels) * 0.4), 8)) 
    y_pos = np.arange(len(piece_labels))
    
    ax.barh(y_pos, attention_values, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(piece_labels)
    ax.invert_yaxis() 
    ax.set_xlabel('Attention Weight', fontsize=12)
    ax.set_ylabel('Pieces', fontsize=12)
    ax.set_title(f'Attention from Square {square_label} to Pieces\nFEN: {fen.split(" ")[0]}', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_filename = f"attention_sq_{square_label}_to_pieces.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close(fig)

def plot_all_squares_attention_to_piece(attention_to_piece: np.ndarray,
                                         target_piece_label: str,
                                         fen: str,
                                         square_idx_to_algebraic: List[str]):
    if attention_to_piece.shape != (64,):
        print(f"Error: attention_to_piece must be of shape (64,). Got {attention_to_piece.shape}")
        return

    heatmap_data = attention_to_piece.reshape(8, 8) 

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap_data, cmap='viridis', origin='lower') 

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([chr(ord('a') + i) for i in range(8)]) 
    ax.set_yticklabels([str(i + 1) for i in range(8)])      

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")
    ax.set_title(f"All Squares' Attention to Piece: {target_piece_label}\nFEN: {fen.split(' ')[0]}", fontsize=12)
    
    plt.tight_layout()
    plot_filename = f"attention_all_sq_to_{target_piece_label.replace('/', '_').replace(' ', '_')}.png" # Sanitize filename
    plt.savefig(plot_filename)
    print(f"Heatmap plot saved to {plot_filename}")
    plt.close(fig)


def run_fen_test():
    print("--- Starting Get Attention Weights from FEN Test with Basic Plot ---")

    square_in_features = 12
    piece_in_features = 12
    embed_dim = 128
    num_actions = 4672
    num_heads_attention = 4
    square_gat_heads = 4
    attention_dropout_rate = 0.1

    chess_net = ChessNetwork(
        square_in_features=square_in_features, piece_in_features=piece_in_features,
        embed_dim=embed_dim, num_actions=num_actions, num_heads=num_heads_attention,
        square_gat_heads=square_gat_heads, attention_dropout_rate=attention_dropout_rate
    )

    chess_net.square_gnn = MockSquareGNN(
        in_features=square_in_features, hidden_features=64, out_features=embed_dim, heads=square_gat_heads
    )
    chess_net.piece_gnn = MockPieceGNN(
        in_channels=piece_in_features, hidden_channels=32, out_channels=embed_dim
    )
    chess_net.policy_head = MockPolicyHead(embed_dim, num_actions)
    chess_net.value_head = MockValueHead(embed_dim)

    mock_converter = MockGNNDataConverter(
        square_features_dim=square_in_features, piece_features_dim=piece_in_features
    )

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    mid_game_fen_actual = "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 9" 
    empty_board_fen = "8/8/8/8/8/8/8/8 w - - 0 1"

    test_fens_for_plot = {
        "Start Position": start_fen,
        "Actual Mid Game": mid_game_fen_actual
    }
    
    square_idx_to_algebraic_a1_h8 = [f"{chr(ord('a') + (i % 8))}{ (i // 8) + 1}" for i in range(64)]

    for name, fen in test_fens_for_plot.items():
        print(f"\n--- Testing FEN for Plotting: {name} ---")
        attention_weights, piece_labels, piece_details = get_gnn_data_and_attention(fen, chess_net, mock_converter)

        if attention_weights is not None and piece_labels and piece_details:
            print(f"  Retrieved Attention Weights Shape: {attention_weights.shape}")
            print(f"  Retrieved {len(piece_labels)} Piece Labels. First 5: {piece_labels[:5]}")
            
            square_to_visualize_idx = 0 
            if name == "Actual Mid Game": 
                square_to_visualize_idx = (4-1)*8 + (ord('d') - ord('a')) 
            
            square_to_visualize_label = square_idx_to_algebraic_a1_h8[square_to_visualize_idx]
            
            if square_to_visualize_idx < attention_weights.shape[0]: # Check against num_squares dimension
                # Ensure piece dimension is also valid for slicing
                if attention_weights.shape[1] == len(piece_labels):
                    attention_for_selected_square = attention_weights[square_to_visualize_idx, :].numpy()
                    print(f"  Plotting bar chart for square: {square_to_visualize_label} (index {square_to_visualize_idx})")
                    plot_square_attention_to_pieces(
                        attention_values=attention_for_selected_square,
                        piece_labels=piece_labels,
                        square_label=square_to_visualize_label,
                        fen=fen
                    )
                else:
                    print(f"  Mismatch between attention_weights piece dimension ({attention_weights.shape[1]}) and piece_labels count ({len(piece_labels)}). Skipping bar chart.")
            else:
                print(f"  Selected square index {square_to_visualize_idx} is out of bounds. Skipping bar chart.")

            target_piece_for_heatmap_label = None
            target_piece_idx_in_list = -1

            if name == "Start Position":
                for i, pd_tuple in enumerate(piece_details):
                    if pd_tuple[1] == 'Q' and pd_tuple[2] == 'd1': 
                        target_piece_for_heatmap_label = pd_tuple[0] 
                        target_piece_idx_in_list = i
                        break
                if target_piece_idx_in_list == -1:
                     print("Could not find White Queen Q on d1 for heatmap. Skipping heatmap for Start Position.")

            elif name == "Actual Mid Game":
                for i, pd_tuple in enumerate(piece_details):
                    if pd_tuple[1] == 'q' and pd_tuple[2] == 'd8': 
                        target_piece_for_heatmap_label = pd_tuple[0]
                        target_piece_idx_in_list = i
                        break
                if target_piece_idx_in_list == -1:
                     print("Could not find Black Queen q on d8 for heatmap. Skipping heatmap for Mid Game.")
            
            if target_piece_for_heatmap_label and target_piece_idx_in_list != -1:
                if target_piece_idx_in_list < attention_weights.shape[1]:
                    attention_to_target_piece = attention_weights[:, target_piece_idx_in_list].numpy()
                    print(f"  Plotting heatmap for piece: {target_piece_for_heatmap_label} (index {target_piece_idx_in_list})")
                    plot_all_squares_attention_to_piece(
                        attention_to_piece=attention_to_target_piece,
                        target_piece_label=target_piece_for_heatmap_label,
                        fen=fen,
                        square_idx_to_algebraic=square_idx_to_algebraic_a1_h8 
                    )
                else:
                    print(f"  Target piece index {target_piece_idx_in_list} is out of bounds for attention_weights piece dimension ({attention_weights.shape[1]}). Skipping heatmap.")
            else:
                print(f"  Target piece for heatmap not identified for {name}. Skipping heatmap.")
        elif name == "Empty Board": 
             pass 
        else:
            print(f"  Could not retrieve attention weights or piece labels for {name}. Skipping plots.")
            if name != "Empty Board":
                 assert False, f"Attention weights were None for {name}, but pieces/labels were expected."
        
    print(f"\n--- Testing FEN for Plotting: Empty Board ---")
    attention_weights_empty, piece_labels_empty, _ = get_gnn_data_and_attention(empty_board_fen, chess_net, mock_converter)
    assert attention_weights_empty is None, "Expected None for empty board, but got something."
    assert not piece_labels_empty, "Expected no piece labels for empty board."
    print("  Attention Weights: None, Piece Labels: Empty (Correct for empty board)")

    print("\n--- Get Attention Weights from FEN Test with Basic Plot Completed ---")

if __name__ == '__main__':
    run_fen_test()
