# tests/test_evaluate_agent.py

import unittest
from unittest.mock import patch, MagicMock, mock_open, call, ANY
import torch
import chess
import chess.pgn
import datetime
from pathlib import Path
import sys
import os

# Add the project root directory to sys.path to allow importing evaluate_agent
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now import from evaluate_agent and other project modules
import evaluate_agent
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

# Use a string that can be imported for patching module-level variables
evaluate_agent_module_path = 'evaluate_agent'

# Mock get_paths to return predictable paths for testing
mock_checkpoints_dir = Path("test_mock_checkpoints")
mock_data_dir = Path("test_mock_data")
mock_stockfish_path = "dummy/path/to/stockfish"


@patch(f'{evaluate_agent_module_path}.get_paths', return_value={
    "checkpoints_dir": mock_checkpoints_dir,
    "training_data_dir": mock_data_dir,
    "stockfish_exe_path": mock_stockfish_path
})
@patch(f'{evaluate_agent_module_path}.DEVICE', torch.device('cpu'))
@patch(f'{evaluate_agent_module_path}.MODEL_CHECKPOINT_FILENAME', "dummy_checkpoint.pth.tar")
@patch(f'{evaluate_agent_module_path}.NUM_EVALUATION_GAMES', 2)
@patch(f'{evaluate_agent_module_path}.MCTS_SIMULATIONS_PER_MOVE', 5)
@patch(f'{evaluate_agent_module_path}.STOCKFISH_GO_COMMAND_DEPTH', 1)
@patch(f'{evaluate_agent_module_path}.STOCKFISH_TIMEOUT_SECONDS', 2)
class TestEvaluateAgent(unittest.TestCase):

    def setUp(self):
        # Define expected sub-module arguments for ChessNetwork reconstruction.
        self.mock_config_params = {
            "gnn_input_features": 12,
            "gnn_hidden_features": 256,
            "gnn_output_features": 128,
            "attention_heads": 4,
            "policy_head_out_moves": 4672,
            "stockfish_path": mock_stockfish_path
        }
        self.expected_pgn_save_dir = mock_data_dir / "evaluation_runs"

    @patch(f'{evaluate_agent_module_path}.Path.is_file', return_value=True)
    @patch(f'{evaluate_agent_module_path}.torch.load')
    @patch(f'{evaluate_agent_module_path}.SquareGNN')
    @patch(f'{evaluate_agent_module_path}.PieceGNN')
    @patch(f'{evaluate_agent_module_path}.CrossAttentionModule')
    @patch(f'{evaluate_agent_module_path}.PolicyHead')
    @patch(f'{evaluate_agent_module_path}.ValueHead')
    @patch(f'{evaluate_agent_module_path}.ChessNetwork')
    def test_load_agent_from_checkpoint_reconstructs_correctly(self, MockChessNetwork, MockValueHead, MockPolicyHead,
                                                            MockCrossAttention, MockPieceGNN, MockSquareGNN,
                                                            mock_torch_load, mock_is_file, mock_get_paths_fixture):
        # This test now passes and needs no changes.
        mock_state_dict = {'model_weights': torch.tensor([1.0])}
        mock_torch_load.return_value = {
            'model_state_dict': mock_state_dict,
            'config_params': self.mock_config_params
        }
        mock_chess_net_inst = MockChessNetwork.return_value
        dummy_checkpoint_path = mock_checkpoints_dir / "dummy_checkpoint.pth.tar"
        loaded_model, loaded_config = evaluate_agent.load_agent_from_checkpoint(dummy_checkpoint_path)
        mock_is_file.assert_called_once()
        mock_torch_load.assert_called_once_with(dummy_checkpoint_path, map_location=torch.device('cpu'))
        MockChessNetwork.assert_called_once()
        mock_chess_net_inst.load_state_dict.assert_called_once_with(mock_state_dict)
        self.assertEqual(loaded_model, mock_chess_net_inst)
        self.assertEqual(loaded_config, self.mock_config_params)

    @patch(f'{evaluate_agent_module_path}.Path.is_file', return_value=True)
    @patch(f'{evaluate_agent_module_path}.StockfishCommunicator')
    def test_initialize_stockfish_player(self, MockStockfishComm, mock_path_is_file, mock_get_paths_fixture):
        # This test now passes and needs no changes.
        mock_sf_instance = MockStockfishComm.return_value
        mock_sf_instance.perform_handshake.return_value = True
        sf_comm = evaluate_agent.initialize_stockfish_player(stockfish_exe_path=mock_stockfish_path)
        mock_path_is_file.assert_called_once()
        MockStockfishComm.assert_called_once_with(stockfish_path=mock_stockfish_path)
        mock_sf_instance.perform_handshake.assert_called_once()
        self.assertEqual(sf_comm, mock_sf_instance)

    def test_get_stockfish_move(self, mock_get_paths_fixture):
        # This test now passes and needs no changes.
        mock_sf_comm = MagicMock(spec=StockfishCommunicator)
        mock_sf_comm._send_command = MagicMock()
        mock_sf_comm._raw_uci_command_exchange.return_value = (True, ["readyok"])
        mock_sf_comm._read_output_until.return_value = (True, ["info depth 1 stuff", "bestmove e2e4 ponder e7e5"])
        move_uci = evaluate_agent.get_stockfish_move(mock_sf_comm, chess.STARTING_FEN, 1, 2)
        self.assertEqual(move_uci, "e2e4")

    @patch(f'{evaluate_agent_module_path}.datetime')
    @patch(f'{evaluate_agent_module_path}.play_evaluation_game')
    @patch(f'{evaluate_agent_module_path}.initialize_stockfish_player')
    @patch(f'{evaluate_agent_module_path}.load_agent_from_checkpoint')
    @patch(f'{evaluate_agent_module_path}.open', new_callable=mock_open)
    def test_run_evaluation_full_loop(self, mock_file_open, mock_load_agent_func,
                                      mock_init_sf_player, mock_play_game_func,
                                      mock_datetime_mod, mock_get_paths_fixture):
        # Arrange
        fixed_timestamp = "20250606_191700"
        mock_datetime_mod.now.return_value.strftime.return_value = fixed_timestamp
        
        mock_agent_network = MagicMock(spec=ChessNetwork)
        mock_load_agent_func.return_value = (mock_agent_network, self.mock_config_params)
        
        mock_sf_player_instance = MagicMock(spec=StockfishCommunicator)
        mock_sf_player_instance.close = MagicMock()
        mock_init_sf_player.return_value = mock_sf_player_instance

        mock_play_game_func.side_effect = [
            (1, "[Event \"Game 1\"]... 1-0"),
            (-1, "[Event \"Game 2\"]... 0-1")
        ]
        
        # Act
        with patch(f'{evaluate_agent_module_path}.MCTS'):
             evaluate_agent.run_evaluation()

        # Assert
        self.assertEqual(mock_play_game_func.call_count, 2)

        # FIX: Check the FIRST positional argument (.args[0]) for the boolean flag.
        self.assertTrue(mock_play_game_func.call_args_list[0].args[0]) # agent_is_white in game 1
        self.assertFalse(mock_play_game_func.call_args_list[1].args[0])# agent_is_white in game 2

        mock_sf_player_instance.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()