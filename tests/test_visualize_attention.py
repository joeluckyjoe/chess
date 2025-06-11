#
# File: tests/test_visualize_attention.py
#
import unittest
import os
import subprocess
import torch

from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead

# Configuration
GNN_INPUT_FEATURES = 12
GNN_OUTPUT_FEATURES = 128 # Using a smaller size for faster testing
NUM_ATTENTION_HEADS = 4
POLICY_HEAD_MOVE_CANDIDATES = 4672
TEMP_MODEL_DIR = "test_outputs"
TEMP_MODEL_PATH = os.path.join(TEMP_MODEL_DIR, "temp_model_for_viz_test.pth.tar")

class TestVisualizeAttentionScript(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create and save a temporary dummy model checkpoint once for all tests.
        """
        os.makedirs(TEMP_MODEL_DIR, exist_ok=True)
        device = torch.device("cpu")

        # Instantiate a dummy model
        square_gnn = SquareGNN(GNN_INPUT_FEATURES, GNN_OUTPUT_FEATURES, GNN_OUTPUT_FEATURES)
        piece_gnn = PieceGNN(GNN_INPUT_FEATURES, GNN_OUTPUT_FEATURES, GNN_OUTPUT_FEATURES)
        attention_module = CrossAttentionModule(
            sq_embed_dim=GNN_OUTPUT_FEATURES, pc_embed_dim=GNN_OUTPUT_FEATURES, num_heads=NUM_ATTENTION_HEADS
        )
        policy_head = PolicyHead(GNN_OUTPUT_FEATURES, POLICY_HEAD_MOVE_CANDIDATES)
        value_head = ValueHead(GNN_OUTPUT_FEATURES)

        network = ChessNetwork(
            square_gnn, piece_gnn, attention_module, policy_head, value_head
        ).to(device)
        
        # Save the dummy model state to a checkpoint file
        torch.save({'state_dict': network.state_dict()}, TEMP_MODEL_PATH)
        print(f"Saved temporary model to {TEMP_MODEL_PATH}")


    @classmethod
    def tearDownClass(cls):
        """
        Clean up the temporary model and output files after all tests are done.
        """
        if os.path.exists(TEMP_MODEL_PATH):
            os.remove(TEMP_MODEL_PATH)
        
        test_output_image = os.path.join(TEMP_MODEL_DIR, "test_viz_output.png")
        if os.path.exists(test_output_image):
            os.remove(test_output_image)

    def test_script_runs_successfully(self):
        """
        Tests if the visualize_attention.py script runs as a subprocess without errors.
        """
        output_image_path = os.path.join(TEMP_MODEL_DIR, "test_viz_output.png")
        fen = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2" # A position after 1. Nf3 Nf6

        # --- THIS IS THE FIX ---
        # Run the script as a module (-m) to ensure Python paths are correct.
        command = [
            "python",
            "-m",
            "visualization.visualize_attention", # Use module path
            "--model_path", TEMP_MODEL_PATH,
            "--fen", fen,
            "--output_path", output_image_path
        ]

        # Execute the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True)

        # Check that the script executed successfully
        self.assertEqual(result.returncode, 0, f"Script failed with error:\n{result.stderr}")
        self.assertIn("Successfully loaded model", result.stdout)
        self.assertIn("Attention plot saved", result.stdout)

        # Check that the output file was actually created
        self.assertTrue(os.path.exists(output_image_path), "Visualization script did not create the output image file.")

if __name__ == '__main__':
    unittest.main()