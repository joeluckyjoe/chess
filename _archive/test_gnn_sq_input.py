import unittest
import chess
import numpy as np
from gnn_sq_input import (
    get_all_square_features_array,
    create_adjacency_matrix,
    get_gnn_sq_input,
    NUM_SQUARES
)
from square_features import SquareFeatures, get_square_features # For feature dimension and direct use # For feature dimension

class TestGnnSqInput(unittest.TestCase):

    def test_get_all_square_features_array(self):
        board = chess.Board()
        features_array = get_all_square_features_array(board)

        self.assertIsInstance(features_array, np.ndarray)
        self.assertEqual(features_array.shape, (NUM_SQUARES, SquareFeatures.get_feature_dimension()))
        self.assertEqual(features_array.dtype, np.float32)

        # Check features for a known square, e.g., e2 (pawn)
        e2_idx = chess.E2 # index 12
        e2_features_manual = SquareFeatures.to_vector(get_square_features(board, e2_idx))
        
        self.assertTrue(np.array_equal(features_array[e2_idx], np.array(e2_features_manual, dtype=np.float32)))

    def test_create_adjacency_matrix_fully_connected(self):
        adj_matrix = create_adjacency_matrix("fully_connected")
        self.assertEqual(adj_matrix.shape, (NUM_SQUARES, NUM_SQUARES))
        
        expected_matrix = np.ones((NUM_SQUARES, NUM_SQUARES), dtype=np.float32)
        np.fill_diagonal(expected_matrix, 0)
        
        self.assertTrue(np.array_equal(adj_matrix, expected_matrix))
        self.assertEqual(np.sum(adj_matrix[0]), NUM_SQUARES - 1) # Square 0 connected to all others

    def test_create_adjacency_matrix_8_way(self):
        adj_matrix = create_adjacency_matrix("8_way")
        self.assertEqual(adj_matrix.shape, (NUM_SQUARES, NUM_SQUARES))

        # Test a corner square (A1, index 0)
        # A1 (0,0) should connect to B1(1), A2(8), B2(9)
        a1_idx = chess.A1
        self.assertEqual(adj_matrix[a1_idx, chess.B1], 1.0)
        self.assertEqual(adj_matrix[a1_idx, chess.A2], 1.0)
        self.assertEqual(adj_matrix[a1_idx, chess.B2], 1.0)
        self.assertEqual(np.sum(adj_matrix[a1_idx]), 3) # 3 neighbors

        # Test a center square (E4, index 28)
        # E4 (rank 3, file 4) should have 8 neighbors
        e4_idx = chess.E4
        self.assertEqual(np.sum(adj_matrix[e4_idx]), 8)
        # Check one neighbor, e.g. D4 (index 27)
        self.assertEqual(adj_matrix[e4_idx, chess.D4], 1.0) 
        # Check a non-neighbor, e.g. A1
        self.assertEqual(adj_matrix[e4_idx, chess.A1], 0.0)
        
        # Check symmetry
        self.assertTrue(np.array_equal(adj_matrix, adj_matrix.T))
        # Check no self-loops
        self.assertTrue(np.all(np.diag(adj_matrix) == 0))


    def test_get_gnn_sq_input(self):
        board = chess.Board()
        node_features, adj_matrix = get_gnn_sq_input(board, adjacency_type="8_way")

        self.assertEqual(node_features.shape, (NUM_SQUARES, SquareFeatures.get_feature_dimension()))
        self.assertEqual(adj_matrix.shape, (NUM_SQUARES, NUM_SQUARES))
        self.assertEqual(np.sum(adj_matrix[chess.A1]), 3) # from 8_way

        node_features_fc, adj_matrix_fc = get_gnn_sq_input(board, adjacency_type="fully_connected")
        self.assertEqual(adj_matrix_fc.shape, (NUM_SQUARES, NUM_SQUARES))
        self.assertEqual(np.sum(adj_matrix_fc[chess.A1]), NUM_SQUARES -1 ) # from fully_connected

    def test_invalid_adjacency_type(self):
        with self.assertRaises(ValueError):
            create_adjacency_matrix("invalid_type")


if __name__ == '__main__':
    unittest.main()