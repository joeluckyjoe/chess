# test_gnn_models.py

import unittest
import torch
import chess

# Assuming gnn_data_converter.py is in the same directory or accessible
from gnn_data_converter import convert_to_gnn_input, GNN_INPUT_FEATURE_DIM
from gnn_models import SquareGNN

# Assuming GNNGraph is defined in gnn_data_converter and PieceGNN in gnn_models
# from gnn_data_converter import GNNGraph # Or wherever GNNGraph is defined
from gnn_models import PieceGNN, NUM_PIECE_FEATURES, PIECE_EMBEDDING_DIM # Make sure constants are accessible or redefined here for test clarity

class TestGNNModels(unittest.TestCase):

    def test_square_gnn_forward_pass(self):
        """
        Tests that the SquareGNN can perform a forward pass and produces an
        output with the correct shape.
        """
        # 1. Define model parameters
        in_features = GNN_INPUT_FEATURE_DIM
        hidden_features = 32
        out_features = 64  # The desired embedding size for each square
        heads = 4

        # 2. Instantiate the model
        model = SquareGNN(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            heads=heads
        )
        model.eval() # Set model to evaluation mode for consistent dropout behavior

        # 3. Create sample input data from a chess board
        board = chess.Board()
        gnn_input = convert_to_gnn_input(board)
        square_graph_data = gnn_input.square_graph

        # 4. Perform the forward pass
        with torch.no_grad(): # Disable gradient calculation for inference
            output_embeddings = model(square_graph_data.x, square_graph_data.edge_index)

        # 5. Assert the output shape is correct
        self.assertIsNotNone(output_embeddings)
        self.assertEqual(output_embeddings.shape, (64, out_features))
        print("\nTestSquareGNN: Forward pass successful.")
        print(f"TestSquareGNN: Output shape is {output_embeddings.shape}, which is correct (64, {out_features}).")

# For testing purposes, let's ensure the constants from gnn_models are used.
# If you change them in gnn_models.py, these tests will use the updated values.

class TestPieceGNN(unittest.TestCase):

    def test_piece_gnn_basic_forward_pass(self):
        model = PieceGNN(in_channels=NUM_PIECE_FEATURES, out_channels=PIECE_EMBEDDING_DIM)
        model.eval() # Set to evaluation mode

        # Example: 10 pieces, each with NUM_PIECE_FEATURES features
        num_pieces = 10
        x_piece = torch.randn(num_pieces, NUM_PIECE_FEATURES)

        # Example edge_index for pieces (e.g., a few connections)
        # Piece 0 connected to 1, 1 to 2, etc., and some other random connections
        edge_index_piece = torch.tensor([
            [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 0, 3, 5],
            [1, 0, 2, 1, 4, 3, 6, 5, 8, 7, 2, 7, 9]
        ], dtype=torch.long)

        # Filter edges to ensure they are within the bounds of num_pieces
        edge_index_piece = edge_index_piece[:, edge_index_piece.max(dim=0).values < num_pieces]


        with torch.no_grad():
            output_embeddings = model(x_piece, edge_index_piece)

        self.assertIsNotNone(output_embeddings)
        self.assertEqual(output_embeddings.shape, (num_pieces, PIECE_EMBEDDING_DIM))

    def test_piece_gnn_variable_piece_count(self):
        model = PieceGNN(in_channels=NUM_PIECE_FEATURES, out_channels=PIECE_EMBEDDING_DIM)
        model.eval()

        # Test with a different number of pieces (e.g., 4 pieces)
        num_pieces_small = 4
        x_piece_small = torch.randn(num_pieces_small, NUM_PIECE_FEATURES)
        # Edges: 0-1, 1-2, 2-3
        edge_index_small = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)

        with torch.no_grad():
            output_small = model(x_piece_small, edge_index_small)

        self.assertEqual(output_small.shape, (num_pieces_small, PIECE_EMBEDDING_DIM))

        # Test with more pieces (e.g., 32 pieces - max pieces)
        num_pieces_large = 32
        x_piece_large = torch.randn(num_pieces_large, NUM_PIECE_FEATURES)
        # Simple chain of edges for testing, adapt if your converter creates denser graphs
        edges_large_src = list(range(num_pieces_large - 1))
        edges_large_dst = list(range(1, num_pieces_large))
        edge_index_large = torch.tensor([edges_large_src + edges_large_dst,
                                         edges_large_dst + edges_large_src], dtype=torch.long)
        if edge_index_large.numel() == 0 and num_pieces_large > 1: # if only one piece, no edges
             edge_index_large = torch.empty((2,0), dtype=torch.long) # PyG expects 2x0 for no edges
        elif num_pieces_large == 1: # Single piece, no edges
            edge_index_large = torch.empty((2,0), dtype=torch.long)


        with torch.no_grad():
            output_large = model(x_piece_large, edge_index_large)

        self.assertEqual(output_large.shape, (num_pieces_large, PIECE_EMBEDDING_DIM))


    def test_piece_gnn_no_pieces(self):
        model = PieceGNN(in_channels=NUM_PIECE_FEATURES, out_channels=PIECE_EMBEDDING_DIM)
        model.eval()

        num_pieces = 0
        x_piece = torch.randn(num_pieces, NUM_PIECE_FEATURES) # This will be a tensor of shape [0, NUM_PIECE_FEATURES]
        edge_index_piece = torch.empty((2, 0), dtype=torch.long) # No edges

        with torch.no_grad():
            output_embeddings = model(x_piece, edge_index_piece)

        self.assertIsNotNone(output_embeddings)
        self.assertEqual(output_embeddings.shape, (0, PIECE_EMBEDDING_DIM))
        self.assertTrue(output_embeddings.device == torch.device('cpu')) # Default device

    def test_piece_gnn_one_piece(self):
        model = PieceGNN(in_channels=NUM_PIECE_FEATURES, out_channels=PIECE_EMBEDDING_DIM)
        model.eval()

        num_pieces = 1
        x_piece = torch.randn(num_pieces, NUM_PIECE_FEATURES)
        edge_index_piece = torch.empty((2,0), dtype=torch.long) # No edges for a single node

        with torch.no_grad():
            output_embeddings = model(x_piece, edge_index_piece)

        self.assertIsNotNone(output_embeddings)
        self.assertEqual(output_embeddings.shape, (num_pieces, PIECE_EMBEDDING_DIM))

    def test_piece_gnn_no_edges(self):
        model = PieceGNN(in_channels=NUM_PIECE_FEATURES, out_channels=PIECE_EMBEDDING_DIM)
        model.eval()

        num_pieces = 5
        x_piece = torch.randn(num_pieces, NUM_PIECE_FEATURES)
        edge_index_piece = torch.empty((2,0), dtype=torch.long) # 5 pieces but no connections

        with torch.no_grad():
            output_embeddings = model(x_piece, edge_index_piece)

        self.assertIsNotNone(output_embeddings)
        self.assertEqual(output_embeddings.shape, (num_pieces, PIECE_EMBEDDING_DIM))
        # In GCN, if there are no edges, the output is typically a transformation of input features.
        # Here, conv1(x, no_edges) -> x', then conv2(x', no_edges) -> x''.
        # We are just checking the shape and that it runs.

# This allows running the tests directly
# if __name__ == '__main__':
#     unittest.main()
if __name__ == '__main__':
    unittest.main()