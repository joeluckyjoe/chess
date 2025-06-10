# gnn_agent/rl_loop/training_supervisor.py

import collections
import logging

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

        Args:
            config (dict): A configuration dictionary containing parameters for the supervisor.
                           Expected keys:
                           - 'supervisor_loss_history_size': Max number of recent loss values to store.
                           - 'bcp_threshold': The probability threshold for Bayesian Change Point detection.
        """
        self.config = config
        self.loss_history_size = self.config.get('supervisor_loss_history_size', 200) # Default to 200 games
        self.bcp_threshold = self.config.get('bcp_threshold', 0.8) # Default high-confidence threshold

        # Use a deque to automatically manage the size of the loss history
        self.self_play_policy_losses = collections.deque(maxlen=self.loss_history_size)

        logging.info("TrainingSupervisor initialized.")
        logging.info(f"Loss history size set to: {self.loss_history_size}")
        logging.info(f"Bayesian Change Point detection threshold set to: {self.bcp_threshold}")

    def record_self_play_loss(self, policy_loss):
        """
        Records a new policy loss from a self-play game.
        """
        self.self_play_policy_losses.append(policy_loss)

    def should_switch_to_mentor(self):
        """
        Analyzes the history of self-play policy losses to detect stagnation.

        This will eventually use a Bayesian Change Point detection model.
        For now, it's a placeholder.

        Returns:
            bool: True if the agent appears to be stagnating and needs mentor guidance.
        """
        # Placeholder logic: For now, we never switch automatically.
        # In the future, this will contain the Bayesian Change Point analysis.
        if len(self.self_play_policy_losses) < self.loss_history_size:
            # Not enough data to make a decision yet
            return False

        logging.info("Analyzing self-play losses for stagnation...")
        # --- FUTURE IMPLEMENTATION ---
        # 1. Convert self.self_play_policy_losses to a numpy array.
        # 2. Run the Bayesian Change Point detection algorithm on the array.
        # 3. If the probability of a recent change point (stagnation) > self.bcp_threshold:
        #    logging.info(f"Stagnation detected! Switching to mentor play.")
        #    return True
        # ---------------------------
        return False

    def should_switch_to_self_play(self, last_mentor_game_result):
        """
        Analyzes the result of the last mentor game to see if the agent has improved.

        Args:
            last_mentor_game_result (dict): A dictionary with metrics from the last mentor game.
                                            e.g., {'outcome': 0.5, 'game_length': 80} (0.5 for draw)

        Returns:
            bool: True if the agent shows improvement and should return to self-play.
        """
        # Placeholder logic: For now, we never switch back automatically.
        # In the future, this will check for draws/wins or improved quality of loss.
        logging.info(f"Analyzing mentor game result: {last_mentor_game_result}")

        # --- FUTURE IMPLEMENTATION ---
        # Example logic: Switch back if the agent achieves a draw or a win.
        # outcome = last_mentor_game_result.get('outcome', 0) # 1 for win, 0.5 for draw, 0 for loss
        # if outcome >= 0.5:
        #    logging.info("Agent achieved a draw or win against mentor. Switching back to self-play.")
        #    return True
        # ---------------------------
        return False