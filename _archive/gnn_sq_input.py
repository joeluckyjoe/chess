import chess
import numpy as np
from typing import Tuple, List

# Assuming square_features.py is in the same directory or accessible in PYTHONPATH
from square_features import get_square_features, SquareFeatures

NUM_SQUARES = 64

def get_all_square_features_array(board: chess.Board) -> np.ndarray:
    """
    Generates a NumPy array containing feature vectors for all 64 squares.
    The order is A1, B1, ..., H1, A2, ..., H8.

    Args:
        board: The python-chess board object.

    Returns:
        A NumPy array of shape (NUM_SQUARES, feature_dimension).
    """
    all_features = []
    for square_index in range(NUM_SQUARES):
        sq_features = get_square_features(board, chess.SQUARES[square_index])
        all_features.append(sq_features.to_vector())
    return np.array(all_features, dtype=np.float32)

def create_adjacency_matrix(adjacency_type: str = "8_way") -> np.ndarray:
    """
    Creates a fixed adjacency matrix for the 64 squares on a chessboard.

    Args:
        adjacency_type: Type of adjacency to create.
                        "8_way": Connects a square to its 8 direct neighbors (horizontal, vertical, diagonal).
                        "fully_connected": Connects every square to every other square.

    Returns:
        A NumPy array of shape (NUM_SQUARES, NUM_SQUARES) representing the adjacency matrix.
        adj[i][j] = 1 if connected, 0 otherwise.
    """
    adj_matrix = np.zeros((NUM_SQUARES, NUM_SQUARES), dtype=np.float32)

    if adjacency_type == "fully_connected":
        adj_matrix.fill(1.0)
        np.fill_diagonal(adj_matrix, 0) # Typically, no self-loops unless specified
        return adj_matrix

    if adjacency_type == "8_way":
        for sq_idx in range(NUM_SQUARES):
            rank = chess.square_rank(sq_idx)
            file = chess.square_file(sq_idx)

            # Iterate over all possible 8 directions
            for dr in [-1, 0, 1]:      # Change in rank
                for df in [-1, 0, 1]:  # Change in file
                    if dr == 0 and df == 0:
                        continue # Skip self

                    new_rank, new_file = rank + dr, file + df

                    if 0 <= new_rank < 8 and 0 <= new_file < 8:
                        neighbor_sq_idx = chess.square(new_file, new_rank)
                        adj_matrix[sq_idx, neighbor_sq_idx] = 1.0
                        adj_matrix[neighbor_sq_idx, sq_idx] = 1.0 # Assuming undirected graph
        return adj_matrix
    
    raise ValueError(f"Unknown adjacency_type: {adjacency_type}")


def get_gnn_sq_input(board: chess.Board, adjacency_type: str = "8_way") -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares the input for the Square-based GNN (G_sq).

    Args:
        board: The python-chess board object.
        adjacency_type: The type of adjacency matrix to create ("8_way" or "fully_connected").

    Returns:
        A tuple containing:
            - node_features: A NumPy array of shape (NUM_SQUARES, feature_dimension)
            - adj_matrix: A NumPy array of shape (NUM_SQUARES, NUM_SQUARES)
    """
    node_features = get_all_square_features_array(board)
    adj_matrix = create_adjacency_matrix(adjacency_type)
    return node_features, adj_matrix