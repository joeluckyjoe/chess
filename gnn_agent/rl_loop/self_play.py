import chess
import torch
from typing import List, Tuple, Dict, Any
import time
import chess.pgn
from datetime import datetime
import random

from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

class SelfPlay:
    """
    Orchestrates a single game of self-play between two MCTS agents,
    generating training data and a PGN of the game.
    """
    def __init__(self, mcts_white: MCTS, mcts_black: MCTS, stockfish_path: str, num_simulations: int, temperature: float = 1.0, temp_decay_moves: int = 30, print_move_timers: bool = False):
        """
        Initializes a self-play game.
        """
        self.mcts_white = mcts_white
        self.mcts_black = mcts_black
        self.game = StockfishCommunicator(stockfish_path)
        self.game.perform_handshake()
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.temp_decay_moves = temp_decay_moves
        # Force timers on for debugging
        self.print_move_timers = True

    def play_game(self) -> Tuple[List[Tuple[Any, Dict[chess.Move, float], float]], chess.pgn.Game]:
        """
        Plays a full game, returning the generated training data and the PGN object.
        """
        print("--- [DEBUG] Self-Play Game Starting ---")
        self.game.reset_board()

        game_history: List[Tuple[Any, Dict[chess.Move, float], bool]] = []
        training_data: List[Tuple[Any, Dict[chess.Move, float], float]] = []

        move_count = 0

        # Modified loop for debugging
        while True:
            move_count += 1
            player_color_str = "White" if self.game.board.turn == chess.WHITE else "Black"
            print(f"\n[DEBUG] Move {move_count} ({player_color_str}): Top of main loop.")

            # --- 1. Check Game Over Status ---
            print(f"[DEBUG] Move {move_count}: Checking game over status with Stockfish...")
            is_over = self.game.is_game_over()
            print(f"[DEBUG] Move {move_count}: Game over status received: {is_over}")
            if is_over:
                print("[DEBUG] Game is over. Breaking loop.")
                break

            # --- 2. Run MCTS Search ---
            loop_start_time = time.time()
            current_player_mcts = self.mcts_white if self.game.board.turn == chess.WHITE else self.mcts_black
            turn_before_move = self.game.board.turn

            mcts_start_time = time.time()
            print(f"[DEBUG] Move {move_count}: Starting MCTS search ({self.num_simulations} simulations)...")
            
            policy, _, board_tensor = current_player_mcts.run_search(
                self.game.board.copy(),
                self.num_simulations
            )
            
            mcts_duration = time.time() - mcts_start_time
            print(f"[DEBUG] Move {move_count}: MCTS search finished.")

            # --- 3. Select and Play Move ---
            move_to_play = None
            if policy:
                print(f"[DEBUG] Move {move_count}: Policy generated with {len(policy)} moves. Selecting move...")
                moves = list(policy.keys())
                move_probs = list(policy.values())

                if move_count <= self.temp_decay_moves:
                    move_to_play = random.choices(moves, weights=move_probs, k=1)[0]
                else:
                    best_move_index = move_probs.index(max(move_probs))
                    move_to_play = moves[best_move_index]
                print(f"[DEBUG] Move {move_count}: Selected move: {move_to_play.uci() if move_to_play else 'None'}")
            else:
                print(f"[DEBUG] Move {move_count}: MCTS returned an empty policy.")


            if move_to_play is None:
                print(f"[INFO] Move {move_count}: No move could be selected. Ending game early.")
                break

            # --- 4. Record and Make Move ---
            game_history.append((board_tensor, policy, turn_before_move))
            print(f"[DEBUG] Move {move_count}: Pushing move '{move_to_play.uci()}' to board...")
            self.game.make_move(move_to_play.uci())
            print(f"[DEBUG] Move {move_count}: Move made successfully.")

            loop_duration = time.time() - loop_start_time
            print(f"[TIMER] Move {move_count}: MCTS search took {mcts_duration:.4f}s. Full loop took {loop_duration:.4f}s.")

        # --- Game is Over ---
        print("\n--- [DEBUG] Post-Game Processing ---")
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

        # --- Generate PGN ---
        pgn = None
        try:
            pgn = chess.pgn.Game()
            pgn.headers["Event"] = "Self-Play Training Game"
            pgn.headers["Site"] = "Colab Environment"
            pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            pgn.headers["White"] = "MCTS_Agent"
            pgn.headers["Black"] = "MCTS_Agent"
            pgn.headers["Result"] = self.game.board.result(claim_draw=True)

            if self.game.board.move_stack:
                node = pgn.add_main_variation(self.game.board.move_stack[0])
                for i in range(1, len(self.game.board.move_stack)):
                    node = node.add_main_variation(self.game.board.move_stack[i])
        except Exception as e:
            print(f"[ERROR] Could not generate PGN for the game: {e}")
        
        # --- RETURN BOTH TRAINING DATA AND PGN OBJECT ---
        return training_data, pgn

    def close(self):
        """Closes the Stockfish process."""
        self.game.close()