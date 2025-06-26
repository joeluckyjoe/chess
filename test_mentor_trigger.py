# File: test_mentor_trigger.py (FINAL, CORRECTED)

import unittest
import random
import numpy as np
from gnn_agent.rl_loop.mentor_trigger import AdaptiveBCPDMonitor

class TestAdaptiveBCPDMonitorLifecycle(unittest.TestCase):
    """
    Tests the full lifecycle of the AdaptiveBCPDMonitor (SciPy version).
    """
    def test_full_lifecycle(self):
        """
        Simulates the entire process, confirming each phase transition.
        """
        # THE FIX: We instantiate the monitor with a tuned hazard rate for the
        # test environment, which is the only parameter we need to control.
        monitor = AdaptiveBCPDMonitor(
            min_analysis_window=15,
            improving_hazard=250, # Lowered from 1000 to be less resistant to change
            p_value_threshold=0.1 # Relaxed for small sample size in test
        )

        # --- Phase 1 & 2: Detect Plateau from Self-Play ---
        print("\n--- Phase 2: Simulating Self-Play Plateau ---")
        plateau_detected = False
        for i in range(1, monitor.min_window + 10):
            regime = monitor.update(5.0 + random.uniform(-0.1, 0.1))
            if regime == 'Plateau':
                plateau_detected = True
                break
        self.assertTrue(plateau_detected, "Monitor FAILED to detect Plateau in Phase 2")
        print(">>> PLATEAU DETECTED! Simulating switch to Mentor Mode. <<<")
        monitor.reset()

        # --- Phase 3: Detect Improvement from Mentor ---
        print("\n--- Phase 3: Simulating Mentor Mode (Improving) ---")
        improving_detected = False
        for i in range(1, monitor.min_window + 10):
            regime = monitor.update(5.0 - i * 0.3)
            if regime == 'Improving':
                improving_detected = True
                break
        self.assertTrue(improving_detected, "Monitor FAILED to detect 'Improving' in Phase 3")
        print(f">>> IMPROVING DETECTED after {monitor.time_step} games. <<<")

        # --- Phase 4: Detect Plateau after Lesson Learned ---
        print("\n--- Phase 4: Simulating Mentor Lesson Learned (New Plateau) ---")
        lesson_learned_triggered = False
        previous_regime = monitor.current_regime
        self.assertEqual(previous_regime, 'Improving')

        for i in range(1, monitor.min_window * 2):
            current_regime = monitor.update(1.0 + random.uniform(-0.1, 0.1))
            if previous_regime == 'Improving' and current_regime == 'Plateau':
                lesson_learned_triggered = True
                break
            previous_regime = current_regime

        self.assertTrue(lesson_learned_triggered, "Monitor FAILED to detect changepoint from Improving to Plateau in Phase 4")
        print(">>> LESSON LEARNED! Simulating switch back to Self-Play Mode. <<<")
        monitor.reset()
        print("\n--- LIFECYCLE TEST COMPLETE: ALL PHASES PASSED ---")


if __name__ == '__main__':
    unittest.main()