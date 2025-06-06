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

@patch(f'{evaluate_agent_module_path}.get_paths', return_value=(mock_checkpoints_dir, mock_data_dir))
@patch(f'{evaluate_agent_module_path}.DEVICE', torch.device('cpu'))
@patch(f'{evaluate_agent_module_path}.MODEL_CHECKPOINT_FILENAME', "dummy_checkpoint.pth.tar")
@patch(f'{evaluate_agent_module_path}.NUM_EVALUATION_GAMES', 2)
@patch(f'{evaluate_agent_module_path}.MCTS_SIMULATIONS_PER_MOVE', 5)
@patch(f'{evaluate_agent_module_path}.STOCKFISH_GO_COMMAND_DEPTH', 1)
@patch(f'{evaluate_agent_module_path}.STOCKFISH_TIMEOUT_SECONDS', 2)
class TestEvaluateAgent(unittest.TestCase):

    def setUp(self):
        # Define expected sub-module arguments for ChessNetwork reconstruction.
        # These should mirror the structure of the 'config' dict in run_training.py
        self.mock_config_params = {
            "gnn_input_features": 12,
            "gnn_hidden_features": 256,
            "gnn_output_features": 128,
            "attention_heads": 4,
            "policy_head_out_moves": 4672,
            "stockfish_path": "dummy/path/to/stockfish" # For testing config loading
        }

    @patch(f'{evaluate_agent_module_path}.torch.load')
    @patch(f'{evaluate_agent_module_path}.SquareGNN')
    @patch(f'{evaluate_agent_module_path}.PieceGNN')
    @patch(f'{evaluate_agent_module_path}.CrossAttentionModule')
    @patch(f'{evaluate_agent_module_path}.PolicyHead')
    @patch(f'{evaluate_agent_module_path}.ValueHead')
    @patch(f'{evaluate_agent_module_path}.ChessNetwork')
    def test_load_agent_from_checkpoint_reconstructs_correctly(self, MockChessNetwork, MockValueHead, MockPolicyHead,
                                                            MockCrossAttention, MockPieceGNN, MockSquareGNN, mock_torch_load,
                                                            mock_get_paths_fixture):
        # Arrange
        mock_state_dict = {'model_weights': torch.tensor([1.0])}
        mock_torch_load.return_value = {
            'model_state_dict': mock_state_dict,
            'config_params': self.mock_config_params
        }

        mock_sq_gnn_inst = MockSquareGNN.return_value
        mock_pc_gnn_inst = MockPieceGNN.return_value
        mock_attn_inst = MockCrossAttention.return_value
        mock_pol_inst = MockPolicyHead.return_value
        mock_val_inst = MockValueHead.return_value
        mock_chess_net_inst = MockChessNetwork.return_value

        dummy_checkpoint_path = mock_checkpoints_dir / "dummy_checkpoint.pth.tar"
        
        # Act
        loaded_model, loaded_config = evaluate_agent.load_agent_from_checkpoint(dummy_checkpoint_path)

        # Assert
        # 1. Check that the checkpoint was loaded
        mock_torch_load.assert_called_once_with(dummy_checkpoint_path, map_location=torch.device('cpu'))
        
        # 2. Check that each sub-module was instantiated with correct args from the config
        MockSquareGNN.assert_called_once_with(
            in_features=self.mock_config_params["gnn_input_features"],
            hidden_features=self.mock_config_params["gnn_hidden_features"],
            out_features=self.mock_config_params["gnn_output_features"],
            heads=self.mock_config_params["attention_heads"]
        )
        MockPieceGNN.assert_called_once_with(
            in_channels=self.mock_config_params["gnn_input_features"],
            hidden_channels=self.mock_config_params["gnn_hidden_features"],
            out_channels=self.mock_config_params["gnn_output_features"]
        )
        MockCrossAttention.assert_called_once_with(
            sq_embed_dim=self.mock_config_params["gnn_output_features"],
            pc_embed_dim=self.mock_config_params["gnn_output_features"],
            num_heads=self.mock_config_params["attention_heads"]
        )
        MockPolicyHead.assert_called_once_with(
            embedding_dim=self.mock_config_params["gnn_output_features"],
            num_possible_moves=self.mock_config_params["policy_head_out_moves"]
        )
        MockValueHead.assert_called_once_with(embedding_dim=self.mock_config_params["gnn_output_features"])
        
        # 3. Check that the main network was instantiated with the created sub-modules
        MockChessNetwork.assert_called_once_with(
            square_gnn=mock_sq_gnn_inst,
            piece_gnn=mock_pc_gnn_inst,
            cross_attention=mock_attn_inst,
            policy_head=mock_pol_inst,
            value_head=mock_val_inst
        )

        # 4. Check that the state dict was loaded and model was set to eval mode
        mock_chess_net_inst.load_state_dict.assert_called_once_with(mock_state_dict)
        mock_chess_net_inst.to.assert_called_once_with(torch.device('cpu'))
        mock_chess_net_inst.eval.assert_called_once()

        # 5. Check return values
        self.assertEqual(loaded_model, mock_chess_net_inst)
        self.assertEqual(loaded_config, self.mock_config_params)


    @patch(f'{evaluate_agent_module_path}.Path.is_file', return_value=True)
    @patch(f'{evaluate_agent_module_path}.os.access', return_value=True)
    @patch(f'{evaluate_agent_module_path}.StockfishCommunicator')
    def test_initialize_stockfish_player(self, MockStockfishComm, mock_os_access, mock_path_is_file, mock_get_paths_fixture):
        # Arrange
        mock_sf_instance = MockStockfishComm.return_value
        mock_sf_instance.perform_handshake.return_value = True # Simulate successful handshake

        # Act
        sf_comm = evaluate_agent.initialize_stockfish_player(stockfish_exe_path="dummy/path/to/stockfish")

        # Assert
        mock_path_is_file.assert_called_once()
        mock_os_access.assert_called_once_with("dummy/path/to/stockfish", os.X_OK)
        MockStockfishComm.assert_called_once_with(stockfish_path="dummy/path/to/stockfish")
        mock_sf_instance.perform_handshake.assert_called_once()
        self.assertEqual(sf_comm, mock_sf_instance)


    def test_get_stockfish_move(self, mock_get_paths_fixture):
        # Arrange
        mock_sf_comm = MagicMock(spec=StockfishCommunicator)
        mock_sf_comm._send_command = MagicMock()
        mock_sf_comm._raw_uci_command_exchange.return_value = (True, ["readyok"]) 
        mock_sf_comm._read_output_until.return_value = (True, ["info depth 1 stuff", "bestmove e2e4 ponder e7e5"])

        board_fen = chess.STARTING_FEN
        depth = 1
        timeout = 2

        # Act
        move_uci = evaluate_agent.get_stockfish_move(mock_sf_comm, board_fen, depth, timeout)

        # Assert
        expected_send_calls = [
            call(f"position fen {board_fen}"),
            call(f"go depth {depth}")
        ]
        mock_sf_comm._send_command.assert_has_calls(expected_send_calls)
        mock_sf_comm._raw_uci_command_exchange.assert_called_once_with("isready", "readyok", timeout=timeout)
        mock_sf_comm._read_output_until.assert_called_once_with("bestmove", timeout=timeout)
        self.assertEqual(move_uci, "e2e4")


    @patch(f'{evaluate_agent_module_path}.datetime')
    @patch(f'{evaluate_agent_module_path}.Path.mkdir')
    @patch(f'{evaluate_agent_module_path}.load_agent_from_checkpoint')
    @patch(f'{evaluate_agent_module_path}.initialize_stockfish_player')
    @patch(f'{evaluate_agent_module_path}.MCTS')
    @patch(f'{evaluate_agent_module_path}.play_evaluation_game')
    @patch(f'{evaluate_agent_module_path}.open', new_callable=mock_open)
    def test_run_evaluation_full_loop(self, mock_file_open, mock_play_game_func, MockMCTSActual,
                                     mock_init_sf_player, mock_load_agent_func, mock_path_mkdir,
                                     mock_datetime_mod, mock_get_paths_fixture):
        # Arrange
        mock_datetime_mod.datetime.now.return_value.strftime.return_value = self.fixed_timestamp
        
        mock_load_agent_func.return_value = (MagicMock(spec=ChessNetwork), self.mock_config_params)
        mock_mcts_instance = MockMCTSActual.return_value
        
        mock_sf_player_instance = MagicMock(spec=StockfishCommunicator)
        mock_sf_player_instance.close = MagicMock()
        mock_init_sf_player.return_value = mock_sf_player_instance

        mock_play_game_func.side_effect = [1, -1] # Agent wins, then loses

        # Act
        evaluate_agent.run_evaluation()

        # Assert
        mock_path_mkdir.assert_called_with(parents=True, exist_ok=True)
        mock_load_agent_func.assert_called_once()
        mock_init_sf_player.assert_called_once_with(self.mock_config_params["stockfish_path"])
        MockMCTSActual.assert_called_once_with(network=mock_load_agent_func.return_value[0], device=ANY)
        
        self.assertEqual(mock_play_game_func.call_count, 2)
        # Check color alternation
        self.assertTrue(mock_play_game_func.call_args_list[0][0][0])  # Game 1: agent_is_white=True
        self.assertFalse(mock_play_game_func.call_args_list[1][0][0]) # Game 2: agent_is_white=False
        
        expected_pgn_file_calls = [
            call(self.expected_pgn_save_dir / "game_001_agent_white.pgn", "w", encoding="utf-8"),
            call(self.expected_pgn_save_dir / "game_002_agent_black.pgn", "w", encoding="utf-8"),
        ]
        mock_file_open.assert_has_calls(expected_pgn_file_calls, any_order=False)
        mock_sf_player_instance.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()