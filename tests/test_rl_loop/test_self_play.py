import unittest
from unittest.mock import MagicMock, patch
import chess
import torch

# Adjust the import path based on your project structure
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

class TestSelfPlay(unittest.TestCase):

    def setUp(self):
        """Set up mock objects for MCTS and StockfishCommunicator."""
        # Mock the MCTS instances
        self.mock_mcts_white = MagicMock(spec=MCTS)
        self.mock_mcts_black = MagicMock(spec=MCTS)

        # Mock the StockfishCommunicator
        # We patch it to avoid actual engine calls
        self.patcher = patch('gnn_agent.rl_loop.self_play.StockfishCommunicator')
        self.mock_stockfish_comm = self.patcher.start()
        
        # Instantiate the mock object to be used in SelfPlay
        self.mock_game_instance = self.mock_stockfish_comm.return_value
        self.mock_game_instance.board = chess.Board()

        # Instantiate SelfPlay with mocks
        self.self_play = SelfPlay(
            mcts_white=self.mock_mcts_white,
            mcts_black=self.mock_mcts_black,
            stockfish_path="dummy_path" # Path is not used due to mocking
        )
        # The SelfPlay __init__ creates its own instance, so we point its 'game' to our mock
        self.self_play.game = self.mock_game_instance

    def tearDown(self):
        """Stop the patcher."""
        self.patcher.stop()

    def test_play_game_loop(self):
        """
        Test that the play_game method correctly simulates a short game
        and generates training data.
        """
        # --- MOCK CONFIGURATION ---
        
        # Define a sequence of game states and outcomes
        # Let's simulate a 3-move game ending in a win for white
        self.mock_game_instance.is_game_over.side_effect = [
            False, # Before White's move
            False, # Before Black's move
            False, # Before White's second move
            True   # Game over
        ]
        
        # White's move 1 (e4)
        move1 = chess.Move.from_uci("e2e4")
        policy1 = {move1: 0.9, chess.Move.from_uci("d2d4"): 0.1}
        tensor1 = torch.randn(1, 10) # Dummy tensor

        # Black's move 1 (e5)
        move2 = chess.Move.from_uci("e7e5")
        policy2 = {move2: 0.8, chess.Move.from_uci("c7c5"): 0.2}
        tensor2 = torch.randn(1, 10)

        # White's move 2 (Nf3)
        move3 = chess.Move.from_uci("g1f3")
        policy3 = {move3: 0.95, chess.Move.from_uci("f1c4"): 0.05}
        tensor3 = torch.randn(1, 10)

        # Configure MCTS mocks to return the predefined moves and policies
        self.mock_mcts_white.run_search.side_effect = [
            (policy1, move1, tensor1),
            (policy3, move3, tensor3)
        ]
        self.mock_mcts_black.run_search.return_value = (policy2, move2, tensor2)

        # Configure the mock board turn correctly
        type(self.mock_game_instance.board).turn = unittest.mock.PropertyMock(
            side_effect=[chess.WHITE, chess.BLACK, chess.WHITE]
        )
        
        # Configure the final game outcome
        self.mock_game_instance.get_game_outcome.return_value = 1.0 # White wins

        # --- EXECUTION ---
        num_simulations = 50
        training_data = self.self_play.play_game(num_simulations)

        # --- ASSERTIONS ---
        
        # 1. Check if MCTS search was called correctly for each player
        self.assertEqual(self.mock_mcts_white.run_search.call_count, 2)
        self.assertEqual(self.mock_mcts_black.run_search.call_count, 1)

        # 2. Check if the moves were made on the board
        self.mock_game_instance.make_move.assert_any_call("e2e4")
        self.mock_game_instance.make_move.assert_any_call("e7e5")
        self.mock_game_instance.make_move.assert_any_call("g1f3")
        self.assertEqual(self.mock_game_instance.make_move.call_count, 3)
        
        # 3. Check the generated training data
        self.assertEqual(len(training_data), 3)
        
        # Verify the first training example (White's first move)
        self.assertTrue(torch.equal(training_data[0][0], tensor1))
        self.assertEqual(training_data[0][1], policy1)
        self.assertEqual(training_data[0][2], 1.0) # Outcome
        
        # Verify the second training example (Black's first move)
        self.assertTrue(torch.equal(training_data[1][0], tensor2))
        self.assertEqual(training_data[1][1], policy2)
        self.assertEqual(training_data[1][2], 1.0) # Outcome

        # Verify the third training example (White's second move)
        self.assertTrue(torch.equal(training_data[2][0], tensor3))
        self.assertEqual(training_data[2][1], policy3)
        self.assertEqual(training_data[2][2], 1.0) # Outcome

if __name__ == '__main__':
    unittest.main()