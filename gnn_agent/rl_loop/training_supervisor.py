# gnn_agent/rl_loop/training_supervisor.py

import collections
import logging
import numpy as np
import ruptures as rpt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingSupervisor:
    """
    Dynamically supervises the training loop, deciding when to switch
    between self-play and mentor-guided play based on performance metrics.
    """

    def __init__(self, config):
        """
        Initializes the TrainingSupervisor.
        """
        self.config = config
        # Stagnation detection parameters
        self.loss_history_size = self.config.get('supervisor_loss_history_size', 200)
        self.stagnation_window = self.config.get('stagnation_window', 0.25)
        self.ruptures_model = self.config.get('ruptures_model', 'l2')
        self.ruptures_penalty = self.config.get('ruptures_penalty', 3)
        self.self_play_policy_losses = collections.deque(maxlen=self.loss_history_size)

        # Improvement detection parameters
        self.mentor_history_size = self.config.get('mentor_history_size', 5)
        self.mentor_win_threshold = self.config.get('mentor_win_threshold', 1)
        self.mentor_draw_threshold = self.config.get('mentor_draw_threshold', 2)
        self.mentor_game_outcomes = collections.deque(maxlen=self.mentor_history_size)

        logging.info("TrainingSupervisor initialized.")
        logging.info(f"Ruptures model set to: {self.ruptures_model}")
        logging.info(f"Mentor game history size set to: {self.mentor_history_size}")

    def _clear_histories(self):
        """Clears all internal histories after a mode switch."""
        logging.info("Clearing supervisor histories after mode switch.")
        self.self_play_policy_losses.clear()
        self.mentor_game_outcomes.clear()

    def record_self_play_loss(self, policy_loss):
        """Records a new policy loss from a self-play game."""
        self.self_play_policy_losses.append(policy_loss)

    def record_mentor_game_outcome(self, outcome):
        """Records a new mentor game outcome (1.0=win, 0.5=draw, 0.0=loss)."""
        self.mentor_game_outcomes.append(outcome)

    def should_switch_to_mentor(self):
        """Analyzes the normalized derivative of the loss history to detect stagnation."""
        if len(self.self_play_policy_losses) < self.loss_history_size:
            return False

        logging.info("Analyzing self-play loss derivative for stagnation...")
        
        loss_array = np.array(self.self_play_policy_losses)
        loss_derivative = np.diff(loss_array)

        std_dev = np.std(loss_derivative)
        if std_dev < 1e-6:
            logging.info("Loss derivative has zero standard deviation. No stagnation detected.")
            return False
        
        normalized_derivative = (loss_derivative - np.mean(loss_derivative)) / std_dev

        try:
            algo = rpt.Pelt(model=self.ruptures_model).fit(normalized_derivative.reshape(-1, 1))
            change_points = algo.predict(pen=self.ruptures_penalty)
        except Exception as e:
            logging.error(f"Ruptures change point detection failed: {e}")
            return False

        if len(change_points) > 1:
            last_change_point = change_points[-2]
            stagnation_start_index = (self.loss_history_size - 1) * (1 - self.stagnation_window)
            
            if last_change_point >= stagnation_start_index:
                logging.warning(f"STAGNATION DETECTED! Change in loss rate found at index {last_change_point}.")
                self._clear_histories() # Clear history before switching mode
                return True

        logging.info("No recent stagnation detected.")
        return False

    def should_switch_to_self_play(self):
        """
        Analyzes the history of mentor games to detect improvement.
        Switches if win/draw thresholds in the recent history are met.
        """
        if len(self.mentor_game_outcomes) < self.mentor_history_size:
            # Not enough mentor games played yet to make a decision
            return False

        wins = list(self.mentor_game_outcomes).count(1.0)
        draws = list(self.mentor_game_outcomes).count(0.5)
        
        logging.info(f"Analyzing mentor game history... Wins: {wins}, Draws: {draws}")

        if wins >= self.mentor_win_threshold or draws >= self.mentor_draw_threshold:
            logging.warning("IMPROVEMENT DETECTED! Agent performance against mentor meets threshold.")
            self._clear_histories() # Clear history before switching mode
            return True
            
        return False