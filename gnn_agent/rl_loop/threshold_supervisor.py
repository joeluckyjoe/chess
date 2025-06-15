import numpy as np
from collections import deque

class ThresholdSupervisor:
    """
    A simple, robust supervisor that switches training mode based on
    statistical thresholds for policy loss (in self-play) and a combination
    of value loss and game length (in mentor-play).
    """
    def __init__(self, config):
        """
        Initializes the supervisor using a configuration dictionary.

        Args:
            config (dict): A dictionary containing all supervisor parameters,
                           typically from the main config.py file.
        """
        self.mode = 'self-play'
        
        # --- Self-Play (Stagnation) Parameters ---
        self.sp_window_size = config.get('SUPERVISOR_WINDOW_SIZE', 20)
        self.volatility_threshold = config.get('SUPERVISOR_VOLATILITY_THRESHOLD', 0.25)
        self.performance_threshold = config.get('SUPERVISOR_PERFORMANCE_THRESHOLD', 1.8)
        self.policy_losses = deque(maxlen=self.sp_window_size)

        # --- Mentor-Play (Graduation) Parameters ---
        self.mp_window_size = config.get('SUPERVISOR_GRADUATION_WINDOW', 10)
        self.graduation_threshold = config.get('SUPERVISOR_GRADUATION_THRESHOLD', 0.05)
        # --- NEW: Add move count threshold ---
        self.move_count_threshold = config.get('SUPERVISOR_MOVE_COUNT_THRESHOLD', 25)
        self.value_losses = deque(maxlen=self.mp_window_size)
        self.move_counts = deque(maxlen=self.mp_window_size)

        print("ThresholdSupervisor initialized.")
        print(f"  - Self-Play Mode: Window={self.sp_window_size}, Volatility > {self.volatility_threshold}, Perf > {self.performance_threshold}")
        print(f"  - Mentor-Play Mode: Window={self.mp_window_size}, Avg Loss < {self.graduation_threshold}, Avg Moves > {self.move_count_threshold}")

    def update(self, metrics):
        """
        Updates the supervisor with new metrics and checks if a mode switch is warranted.
        """
        if self.mode == 'self-play':
            policy_loss = metrics.get('policy_loss')
            if policy_loss is not None:
                self.policy_losses.append(policy_loss)
                self._check_for_stagnation()
        
        elif self.mode == 'mentor-play':
            value_loss = metrics.get('value_loss')
            # --- NEW: Get the number of moves from the metrics ---
            num_moves = metrics.get('num_moves')
            
            if value_loss is not None and num_moves is not None:
                self.value_losses.append(value_loss)
                self.move_counts.append(num_moves)
                self._check_for_graduation()
        
        return self.mode

    def _check_for_stagnation(self):
        """Checks if the agent should switch from self-play to mentor-play."""
        if len(self.policy_losses) < self.sp_window_size:
            return

        window_data = np.array(self.policy_losses)
        
        current_volatility = np.std(window_data)
        if current_volatility > self.volatility_threshold:
            print(f"SUPERVISOR: Volatility threshold breached ({current_volatility:.3f} > {self.volatility_threshold}). Switching to MENTOR mode.")
            self._switch_mode('mentor-play')
            return

        current_performance = np.mean(window_data)
        if current_performance > self.performance_threshold:
            print(f"SUPERVISOR: Performance threshold breached ({current_performance:.3f} > {self.performance_threshold}). Switching to MENTOR mode.")
            self._switch_mode('mentor-play')

    def _check_for_graduation(self):
        """
        Checks if the agent can graduate based on low value loss AND high game length.
        """
        if len(self.value_losses) < self.mp_window_size:
            return

        avg_value_loss = np.mean(np.array(self.value_losses))
        avg_move_count = np.mean(np.array(self.move_counts))

        # --- NEW: Check both conditions ---
        is_loss_good = avg_value_loss < self.graduation_threshold
        is_length_good = avg_move_count > self.move_count_threshold

        if is_loss_good and is_length_good:
            print(f"SUPERVISOR: Graduation thresholds met (Avg Loss: {avg_value_loss:.4f}, Avg Moves: {avg_move_count:.1f}). Switching to SELF-PLAY mode.")
            self._switch_mode('self-play')
        elif is_loss_good and not is_length_good:
             # This is the "cheap win/loss" scenario we want to avoid
            print(f"SUPERVISOR: Value loss is low ({avg_value_loss:.4f}) but avg game length ({avg_move_count:.1f}) is too short. Graduation denied.")


    def _switch_mode(self, new_mode):
        """Switches the mode and resets all internal state."""
        self.mode = new_mode
        self.policy_losses.clear()
        self.value_losses.clear()
        self.move_counts.clear() # --- NEW: Reset move counts ---
        print(f"SUPERVISOR: Mode changed to '{new_mode}'. All internal loss histories have been reset.")
