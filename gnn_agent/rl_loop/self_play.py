import chess
import torch
from typing import List, Tuple, Dict, Any
import time
import chess.pgn
from datetime import datetime
import random # NEW: For temperature sampling

from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

class SelfPlay:
    """
    Orchestrates a single game of self-play between two MCTS agents,
    generating training data.
    """
    # MODIFIED: __init__ to accept temperature parameters
    def __init__(self, mcts_white: MCTS, mcts_black: MCTS, stockfish_path: str, num_simulations: int, temperature: float = 1.0, temp_decay_moves: int = 30):
        """
        Initializes a self-play game.

        Args:
            mcts_white: The MCTS search instance for the white player.
            mcts_black: The MCTS search instance for the black player.
            stockfish_path: Path to the Stockfish executable.
            num_simulations: The number of MCTS simulations to run per move.
            temperature (float): The temperature for sampling moves, encouraging exploration.
            temp_decay_moves (int): The number of moves after which temperature becomes 0 (greedy).
        """
        self.mcts_white = mcts_white
        self.mcts_black = mcts_black
        self.game = StockfishCommunicator(stockfish_path)
        self.game.perform_handshake()
        self.num_simulations = num_simulations
        self.temperature = temperature # NEW
        self.temp_decay_moves = temp_decay_moves # NEW

    def play_game(self) -> List[Tuple[Any, Dict[chess.Move, float], float]]:
        """
        Plays a full game, returning the generated training data.
        This version includes temperature sampling for exploration.
        """
        print("Starting a new self-play game...")
        self.game.reset_board()
        
        game_history: List[Tuple[Any, Dict[chess.Move, float], bool]] = [] 
        training_data: List[Tuple[Any, Dict[chess.Move, float], float]] = []
        
        move_count = 0

        while not self.game.is_game_over():
            move_count += 1
            loop_start_time = time.time()

            current_player_mcts = self.mcts_white if self.game.board.turn == chess.WHITE else self.mcts_black
            turn_before_move = self.game.board.turn

            mcts_start_time = time.time()
            # MCTS search returns the policy (move probabilities) and the raw GNN input
            policy, _, board_tensor = current_player_mcts.run_search(
                self.game.board.copy(),
                self.num_simulations
            )
            mcts_duration = time.time() - mcts_start_time
            
            # --- START: NEW Temperature-based Move Selection ---
            move_to_play = None
            if policy: # Ensure the policy dictionary is not empty
                moves = list(policy.keys())
                move_probs = list(policy.values())

                if move_count <= self.temp_decay_moves:
                    # EXPLORATION: Sample a move based on the policy probabilities
                    # The random.choices function is perfect for weighted sampling
                    move_to_play = random.choices(moves, weights=move_probs, k=1)[0]
                else:
                    # EXPLOITATION: Choose the best move (highest probability)
                    best_move_index = move_probs.index(max(move_probs))
                    move_to_play = moves[best_move_index]
            # --- END: NEW Temperature-based Move Selection ---
            
            if move_to_play is None:
                print(f"[INFO] Move {move_count}: No move could be selected. Ending game early.")
                break
            
            game_history.append((board_tensor, policy, turn_before_move))
            self.game.make_move(move_to_play.uci()) # MODIFIED: Use the move we selected

            loop_duration = time.time() - loop_start_time
            print(f"[TIMER] Move {move_count}: MCTS search took {mcts_duration:.4f}s. Full loop took {loop_duration:.4f}s.")

        # --- Game is Over ---
        
        # 1. Finalize Training Data
        raw_outcome = self.game.get_game_outcome()
        print(f"\nGame over. Raw outcome (White's perspective): {raw_outcome}. Total moves: {len(game_history)}")
        for board_tensor_hist, policy_hist, turn_at_state in game_history:
            value_for_state = 0.0
            if raw_outcome is not None:
                if turn_at_state == chess.WHITE:
                    value_for_state = raw_outcome
                else:
                    value_for_state = -raw_outcome
            training_data.append((board_tensor_hist, policy_hist, value_for_state))

        # 2. Generate and Print PGN
        try:
            pgn = chess.pgn.Game()
            pgn.headers["Event"] = "Self-Play Training Game"
            pgn.headers["Site"] = "Colab Environment"
            pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            pgn.headers["White"] = "MCTS_Agent"
            pgn.headers["Black"] = "MCTS_Agent"
            pgn.headers["Result"] = self.game.board.result(claim_draw=True)
            
            node = pgn.add_main_variation(self.game.board.move_stack[0])
            for i in range(1, len(self.game.board.move_stack)):
                node = node.add_main_variation(self.game.board.move_stack[i])

            print("\n--- PGN START ---")
            print(pgn)
            print("--- PGN END ---\n")
        except Exception as e:
            print(f"[ERROR] Could not generate PGN for the game: {e}")

        return training_data

    def close(self):
        """Closes the Stockfish process."""
        self.game.close()