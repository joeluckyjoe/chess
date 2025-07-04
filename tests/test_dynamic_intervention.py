# FILENAME: tests/test_dynamic_intervention.py

import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import os

# We need to import the function we intend to test
from run_training import main as run_training_main

@pytest.fixture
def mock_dependencies(tmp_path):
    """
    Mocks all major dependencies of the training loop, completely isolating
    it from the file system and other modules like config.py.
    """
    # This dictionary will be used to patch 'run_training.config_params'
    mock_config = {
        'DEVICE': 'cpu', 'STOCKFISH_PATH': '/usr/bin/stockfish', 'CPUCT': 1.25,
        'BATCH_SIZE': 16, 'MCTS_SIMULATIONS': 100, 'MENTOR_ELO_RATING': 1400,
        'MENTOR_GAME_AGENT_COLOR': 'white', 'TOTAL_GAMES': 20,
        'CHECKPOINT_INTERVAL': 10, 'SUPERVISOR_GRACE_PERIOD': 3,
        'MAX_INTERVENTION_GAMES': 3, 'SUPERVISOR_BAYESIAN_PENALTY': 0.8,
        'SUPERVISOR_WINDOW_SIZE': 5, 'SUPERVISOR_PERFORMANCE_THRESHOLD': 7.0,
        'SUPERVISOR_RECENCY_WINDOW': 10, 'PUZZLE_RATIO': 0.25
    }

    # This mock object will be used to patch 'run_training.get_paths'
    # Create the directories that the main script expects to exist.
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "training_data").mkdir()
    (tmp_path / "pgn_games").mkdir()

    mock_paths = MagicMock(
        checkpoints_dir=tmp_path / "checkpoints",
        training_data_dir=tmp_path / "training_data",
        pgn_games_dir=tmp_path / "pgn_games",
        drive_project_root=tmp_path,
        tactical_puzzles_file=tmp_path / "puzzles.jsonl"
    )

    # Use a context manager to handle starting and stopping all patches
    with patch.dict('os.environ', {}, clear=True), \
         patch('run_training.get_paths', return_value=mock_paths) as p_paths, \
         patch('run_training.config_params', mock_config) as p_config, \
         patch('run_training.Trainer') as p_trainer, \
         patch('run_training.MCTS') as p_mcts, \
         patch('run_training.SelfPlay') as p_self_play, \
         patch('run_training.MentorPlay') as p_mentor_play, \
         patch('run_training.TrainingDataManager') as p_data_manager, \
         patch('run_training.BayesianSupervisor') as p_supervisor, \
         patch('run_training.load_tactical_puzzles', return_value=[]) as p_puzzles, \
         patch('argparse.ArgumentParser') as p_argparse:

        # Configure mock return values
        p_argparse.return_value.parse_args.return_value = MagicMock(
            force_start_game=None, disable_puzzle_mixing=True
        )
        p_self_play.return_value.play_game.return_value = (['self_play_example'], 'pgn_self_play')
        p_mentor_play.return_value.play_game.return_value = (['mentor_play_example'], 'pgn_mentor_play')
        p_trainer.return_value.train_on_batch.return_value = (5.0, 0.5)
        p_trainer.return_value.load_or_initialize_network.return_value = (MagicMock(), 0)

        # Yield the necessary mocks and config to the tests
        yield {
            "supervisor": p_supervisor,
            "loss_log_path": tmp_path / 'loss_log_v2.csv',
            "mock_config": mock_config, # Yield the dict so tests can modify it
            "max_intervention_games": mock_config['MAX_INTERVENTION_GAMES'],
            "grace_period": mock_config['SUPERVISOR_GRACE_PERIOD']
        }

def test_intervention_starts_and_stabilizes(mock_dependencies):
    supervisor_mock = mock_dependencies['supervisor'].return_value
    loss_log_path = mock_dependencies['loss_log_path']
    mock_config = mock_dependencies['mock_config']

    supervisor_mock.check_for_stagnation.side_effect = [True, False, False]
    mock_config['TOTAL_GAMES'] = 5

    run_training_main()

    log_df = pd.read_csv(loss_log_path)
    # *** FIX: The first game is mentor-play, followed by a grace period, then a final check.
    expected_game_types = ['mentor-play', 'self-play', 'self-play', 'self-play', 'self-play']
    
    assert log_df['game_type'].tolist() == expected_game_types
    # *** FIX: Supervisor is called on game 1, after game 1, and on game 5.
    assert supervisor_mock.check_for_stagnation.call_count == 3
    assert log_df.shape[0] == 5

def test_intervention_respects_max_game_limit(mock_dependencies):
    supervisor_mock = mock_dependencies['supervisor'].return_value
    loss_log_path = mock_dependencies['loss_log_path']
    mock_config = mock_dependencies['mock_config']
    max_intervention = mock_dependencies['max_intervention_games']

    supervisor_mock.check_for_stagnation.return_value = True
    total_games_to_run = 1 + max_intervention + mock_dependencies['grace_period']
    mock_config['TOTAL_GAMES'] = total_games_to_run

    run_training_main()

    log_df = pd.read_csv(loss_log_path)
    # *** FIX: First game is mentor, then burst continues, then grace period, then one last mentor game.
    expected_types = ['mentor-play'] * max_intervention + ['self-play'] * mock_dependencies['grace_period'] + ['mentor-play']

    assert log_df.shape[0] == total_games_to_run
    assert log_df['game_type'].tolist() == expected_types

def test_grace_period_prevents_supervisor_check(mock_dependencies):
    supervisor_mock = mock_dependencies['supervisor'].return_value
    loss_log_path = mock_dependencies['loss_log_path']
    mock_config = mock_dependencies['mock_config']
    grace_period = mock_dependencies['grace_period']

    supervisor_mock.check_for_stagnation.side_effect = [True, False, False]
    total_games_to_run = 1 + 1 + grace_period
    mock_config['TOTAL_GAMES'] = total_games_to_run

    run_training_main()

    log_df = pd.read_csv(loss_log_path)
    # *** FIX: The first game is mentor-play, followed by a grace period.
    expected_types = ['mentor-play'] + ['self-play'] * (1 + grace_period)

    assert log_df.shape[0] == total_games_to_run
    assert log_df['game_type'].tolist() == expected_types
    # *** FIX: Supervisor is called on game 1, after game 1, and on game 5.
    assert supervisor_mock.check_for_stagnation.call_count == 3