import pytest
from gnn_agent.rl_loop.threshold_supervisor import ThresholdSupervisor

@pytest.fixture
def default_config():
    """Provides a default supervisor configuration for testing."""
    return {
        'SUPERVISOR_WINDOW_SIZE': 20,
        'SUPERVISOR_VOLATILITY_THRESHOLD': 0.25,
        'SUPERVISOR_PERFORMANCE_THRESHOLD': 1.8,
        'SUPERVISOR_GRADUATION_WINDOW': 10,
        'SUPERVISOR_GRADUATION_THRESHOLD': 0.05,
        'SUPERVISOR_MOVE_COUNT_THRESHOLD': 25,
    }

class TestThresholdSupervisor:
    """Unit tests for the ThresholdSupervisor."""

    def test_initialization(self, default_config):
        """Tests correct initialization."""
        supervisor = ThresholdSupervisor(config=default_config)
        assert supervisor.mode == 'self-play'

    def test_switch_on_high_volatility(self, default_config):
        """Tests that a switch occurs on high volatility."""
        supervisor = ThresholdSupervisor(config=default_config)
        for i in range(default_config['SUPERVISOR_WINDOW_SIZE']):
            loss = 0.1 if i % 2 == 0 else 1.5
            supervisor.update({'policy_loss': loss})
        assert supervisor.mode == 'mentor-play'

    def test_switch_on_high_loss(self, default_config):
        """Tests that a switch occurs on high average loss."""
        supervisor = ThresholdSupervisor(config=default_config)
        for _ in range(default_config['SUPERVISOR_WINDOW_SIZE']):
            supervisor.update({'policy_loss': 1.9})
        assert supervisor.mode == 'mentor-play'

    def test_graduation_denied_for_short_games(self, default_config):
        """
        Tests that graduation is DENIED if value loss is low but games
        are too short. This prevents graduating on "cheap wins/losses".
        """
        supervisor = ThresholdSupervisor(config=default_config)
        supervisor.mode = 'mentor-play' # Force into mentor mode

        # Simulate 10 games with great value loss but short game length
        for _ in range(default_config['SUPERVISOR_GRADUATION_WINDOW']):
            metrics = {'value_loss': 0.01, 'num_moves': 15} # Moves are below the 25 threshold
            supervisor.update(metrics)
        
        assert supervisor.mode == 'mentor-play', "Supervisor graduated incorrectly with short games."

    def test_graduation_granted_on_good_performance(self, default_config):
        """
        Tests that graduation is GRANTED only when both value loss is low
        and game length is high.
        """
        supervisor = ThresholdSupervisor(config=default_config)
        supervisor.mode = 'mentor-play' # Force into mentor mode

        # Simulate 10 games with great value loss AND long game length
        for _ in range(default_config['SUPERVISOR_GRADUATION_WINDOW']):
            metrics = {'value_loss': 0.01, 'num_moves': 30} # Both metrics pass their thresholds
            supervisor.update(metrics)
        
        assert supervisor.mode == 'self-play', "Supervisor failed to graduate with valid conditions."
        
        # Also verify that the internal data queues were reset after the switch
        assert len(supervisor.policy_losses) == 0
        assert len(supervisor.value_losses) == 0
        assert len(supervisor.move_counts) == 0

