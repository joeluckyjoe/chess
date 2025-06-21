import unittest
import torch
import chess

# Update imports to reflect the project structure
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.neural_network.chess_network import ChessNetwork

class TestChessNetworkForwardPass(unittest.TestCase):

    def setUp(self):
        """
        Set up the components of the ChessNetwork for testing.
        This follows the current modular architecture.
        """
        # 1. Model Configuration
        # CORRECTED: The data converter provides 12 features per node.
        self.gnn_input_dim = 12
        self.embedding_dim = 64
        self.gnn_hidden_dim = 128
        self.gnn_heads = 4
        self.num_possible_moves = 4672

        # 2. Instantiate all network components with the CORRECT arguments
        self.square_gnn = SquareGNN(
            in_features=self.gnn_input_dim,
            hidden_features=self.gnn_hidden_dim,
            out_features=self.embedding_dim,
            heads=self.gnn_heads
        )
        self.piece_gnn = PieceGNN(
            in_channels=self.gnn_input_dim,
            hidden_channels=self.gnn_hidden_dim,
            out_channels=self.embedding_dim
        )
        self.cross_attention = CrossAttentionModule(
            sq_embed_dim=self.embedding_dim,
            pc_embed_dim=self.embedding_dim,
            num_heads=self.gnn_heads
        )
        self.policy_head = PolicyHead(
            embedding_dim=self.embedding_dim,
            num_possible_moves=self.num_possible_moves
        )
        self.value_head = ValueHead(embedding_dim=self.embedding_dim)


        # 3. Assemble the full ChessNetwork
        self.model = ChessNetwork(
            square_gnn=self.square_gnn,
            piece_gnn=self.piece_gnn,
            cross_attention=self.cross_attention,
            policy_head=self.policy_head,
            value_head=self.value_head
        )
        self.model.eval()

    def _run_pass(self, board: chess.Board, test_name: str):
        """Helper function to run a forward pass and perform checks."""
        print(f"\n--- Running Forward Pass Test: {test_name} ---")
        print(f"Board FEN: {board.fen()}")

        # Create realistic input data
        gnn_input = convert_to_gnn_input(board, device='cpu')
        
        # Unpack GNNInput for the model
        square_features = gnn_input.square_graph.x
        square_edge_index = gnn_input.square_graph.edge_index
        piece_features = gnn_input.piece_graph.x
        piece_edge_index = gnn_input.piece_graph.edge_index
        piece_to_square_map = gnn_input.piece_to_square_map
        
        if piece_features is None:
            # Handle case with no pieces for PieceGNN if converter returns None
            piece_features = torch.empty(0, self.gnn_input_dim, device='cpu')
            piece_edge_index = torch.empty(2, 0, dtype=torch.long, device='cpu')
            piece_to_square_map = torch.empty(0, dtype=torch.long, device='cpu')

        # Perform the forward pass
        with torch.no_grad():
            try:
                policy_logits, value = self.model(
                    square_features,
                    square_edge_index,
                    piece_features,
                    piece_edge_index,
                    piece_to_square_map
                )
            except Exception as e:
                self.fail(f"Forward pass for '{test_name}' failed with an exception: {e}")

        # Check output shapes (unbatched)
        self.assertEqual(policy_logits.shape, (self.num_possible_moves,), f"Policy shape for {test_name} is wrong.")
        self.assertEqual(value.shape, (1,), f"Value shape for {test_name} is wrong.")
        
        # Check value range
        self.assertTrue(-1.0 <= value.item() <= 1.0, f"Value {value.item()} for {test_name} out of range [-1, 1]")
        
        print(f"Test '{test_name}' successful.")
        print(f"Policy output shape: {policy_logits.shape}")
        print(f"Value output shape: {value.shape}, Value: {value.item():.4f}")

    def test_forward_pass_scenarios(self):
        """
        Tests the full forward pass of the ChessNetwork for different board states.
        """
        # Scenario 1: Standard starting position
        start_board = chess.Board()
        self._run_pass(start_board, "Starting Position")

        # Scenario 2: Mid-game position
        mid_game_board = chess.Board("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        self._run_pass(mid_game_board, "Mid-game Position")
        
        # Scenario 3: Position with few pieces (tests robustness)
        end_game_board = chess.Board("8/8/8/4k3/8/3K4/8/8 w - - 0 1")
        self._run_pass(end_game_board, "End-game Position (Kings Only)")

if __name__ == '__main__':
    unittest.main()