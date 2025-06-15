import pytest
import numpy as np
from gnn_agent.rl_loop.bayesian_supervisor import BayesianTrainingSupervisor

@pytest.fixture
def supervisor_config():
    """Provides a standard configuration for the supervisor."""
    # The lambda parameter for bcd.constant_hazard corresponds to the
    # average run length between change points. 250 is a reasonable default.
    return {
        "self_play_config": {'lambda': 250},
        "mentor_play_config": {'lambda': 250}
    }

class TestBayesianTrainingSupervisor:
    """Unit tests for the BayesianTrainingSupervisor."""

    def test_initialization(self, supervisor_config):
        """
        Tests that the supervisor initializes in the correct mode and with
        the correct initial state.
        """
        supervisor = BayesianTrainingSupervisor(**supervisor_config)
        assert supervisor.mode == 'self-play', "Supervisor should start in self-play mode."
        assert supervisor.policy_losses == [], "Policy loss history should be empty on init."
        assert supervisor.value_losses == [], "Value loss history should be empty on init."

    def test_no_switch_on_stable_data(self, supervisor_config):
        """
        Tests that the supervisor remains in self-play mode when fed a stream
        of stable, low-variance data.
        """
        supervisor = BayesianTrainingSupervisor(**supervisor_config)
        
        # Simulate 20 games with stable policy loss
        for i in range(20):
            stable_metrics = {'policy_loss': 0.5 + np.random.normal(0, 0.01)}
            mode = supervisor.update(stable_metrics)
            assert mode == 'self-play', f"Supervisor switched mode prematurely on game {i+1} with stable data."

    def test_switch_to_mentor_on_changepoint(self, supervisor_config):
        """
        Tests that the supervisor correctly switches from self-play to mentor
        mode when a clear change point in the data stream is introduced.
        """
        supervisor = BayesianTrainingSupervisor(**supervisor_config)

        # Phase 1: Simulate 15 games of stable, low policy loss
        for _ in range(15):
            stable_metrics = {'policy_loss': 0.5 + np.random.normal(0, 0.01)}
            supervisor.update(stable_metrics)
        
        assert supervisor.mode == 'self-play', "Supervisor should still be in self-play after stable phase."

        # Phase 2: Introduce a sharp change (stagnation/oscillation)
        # We simulate this with a jump in the mean and variance of policy loss
        print("\n--- Introducing sharp change in metrics ---")
        changed_metric = {'policy_loss': 1.5 + np.random.normal(0, 0.1)}
        mode = supervisor.update(changed_metric)

        assert mode == 'mentor-play', "Supervisor failed to switch to mentor mode after a clear change point."
        assert supervisor.value_losses == [], "Value loss history should be reset after switching to mentor mode."
        assert supervisor.game_lengths == [], "Game length history should be reset after switching to mentor mode."

    def test_ignores_missing_metrics(self, supervisor_config):
        """
        Tests that the supervisor's state does not change if the required
        metrics are missing from the input dictionary.
        """
        supervisor = BayesianTrainingSupervisor(**supervisor_config)
        initial_mode = supervisor.mode
        
        # Update with incomplete metrics
        supervisor.update({'some_other_metric': 1.0})
        
        assert supervisor.mode == initial_mode, "Supervisor mode changed despite missing metrics."
        assert len(supervisor.policy_losses) == 0, "Policy losses should not be updated with incomplete metrics."