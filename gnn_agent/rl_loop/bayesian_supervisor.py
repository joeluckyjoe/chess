import numpy as np
import bayesian_changepoint_detection as bcd
from functools import partial

class BayesianTrainingSupervisor:
    """
    A training supervisor that uses online Bayesian change point detection to
    monitor the agent's learning progress and decide when to switch between
    self-play and mentor-play modes.
    """
    def __init__(self, self_play_config, mentor_play_config):
        """
        Initializes the supervisor with separate BCD models for self-play
        and mentor-play phases.
        """
        self.mode = 'self-play'
        # The 'rate' parameter is the correct one for ifm_hazard
        self.self_play_config = {'rate': self_play_config.get('lambda', 250)}
        self.mentor_play_config = {'rate': mentor_play_config.get('lambda', 250)}
        self.t = 0

        self._reset_self_play_detectors()
        self._reset_mentor_detectors()
        print("BayesianTrainingSupervisor initialized. Starting in SELF-PLAY mode.")

    def update(self, metrics):
        """
        Updates the appropriate detectors with new metrics from the latest game
        and checks if a mode switch is warranted.
        """
        self.t += 1
        
        if self.mode == 'self-play':
            policy_loss = metrics.get('policy_loss')
            if policy_loss is None:
                return self.mode

            # The correct function is detect_online
            self.policy_loss_detector.detect_online(policy_loss)
            
            # Access the probability distribution via the 'changepoint_p' attribute
            if self.policy_loss_detector.changepoint_p[-1] > 0.5 and self.t > 1:
                print(f"SUPERVISOR: Change point detected in policy loss dynamics (game #{self.t}). Switching to MENTOR mode.")
                self.mode = 'mentor-play'
                self._reset_mentor_detectors()

        elif self.mode == 'mentor-play':
            # This part remains the same logic
            value_loss = metrics.get('value_loss')
            game_length = metrics.get('game_length')
            
            if value_loss is None or game_length is None:
                return self.mode

            self.value_loss_detector.detect_online(value_loss)
            self.game_length_detector.detect_online(game_length)
            
            # TODO: Implement robust graduation logic.
            pass

        return self.mode

    def _reset_self_play_detectors(self):
        """Resets the state of the self-play detectors."""
        # The correct hazard function is 'ifm_hazard'
        hazard = partial(bcd.ifm_hazard, **self.self_play_config)
        # We will use a Normal distribution with a known (but wide) sigma
        observation_model = bcd.NormalKnownSigma(prior_sigma=1.0)
        self.policy_loss_detector = bcd.Detector(hazard, observation_model)
        self.t = 0
        print("SUPERVISOR: Self-play detectors have been reset.")

    def _reset_mentor_detectors(self):
        """Resets the state of the mentor-mode detectors."""
        hazard = partial(bcd.ifm_hazard, **self.mentor_play_config)
        observation_model = bcd.NormalKnownSigma(prior_sigma=1.0)
        self.value_loss_detector = bcd.Detector(hazard, observation_model)
        self.game_length_detector = bcd.Detector(hazard, observation_model)
        print("SUPERVISOR: Mentor-mode detectors have been reset.")