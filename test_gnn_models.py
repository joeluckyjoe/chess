# test_gnn_models.py

import unittest
import torch
import chess

# Assuming gnn_data_converter.py is in the same directory or accessible
from gnn_data_converter import convert_to_gnn_input, GNN_INPUT_FEATURE_DIM
from gnn_models import SquareGNN

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


if __name__ == '__main__':
    unittest.main()