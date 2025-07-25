# /home/giuseppe/chess/rl_loop/self_play.py
import chess
import torch
from typing import List, Tuple, Dict
import time
import chess.pgn
from datetime import datetime

# --- MODIFIED: Import the new model ---
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

class SelfPlay:
    """
    Orchestrates a single game of self-play for the ValueNextStateModel.
    """
    # --- MODIFIED: Update type hint for the network ---
    def __init__(self, network: ValueNextStateModel, device: torch.device,
                 mcts_white: MCTS, mcts_black: MCTS, 
                 stockfish_path: str, num_simulations: int, 
                 temperature: float = 1.0, temp_decay_moves: int = 30, 
                 print_move_timers: bool = False, contempt_factor: float = 0.0):
        self.network = network
        self.device = device
        self.mcts_white = mcts_white
        self.mcts_black = mcts_black
        self.game = StockfishCommunicator(stockfish_path)
        self.game.perform_handshake()
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.temp_decay_moves = temp_decay_moves
        self.print_move_timers = print_move_timers
        self.contempt_factor = contempt_factor

    # --- MODIFIED: Update the return type hint ---
    def play_game(self) -> Tuple[List[Tuple[str, Dict[chess.Move, float], float, float]], chess.pgn.Game]:
        """
        Plays a full game, generating training data including next-state values.
        Returns tuples of (FEN, policy, value, next_state_value).
        """
        print("Starting a new self-play game...")
        self.game.reset_board()

        # The history will now store an extra item: the next_state_value
        game_history: List[Tuple[str, Dict[chess.Move, float], bool, float]] = []
        move_count = 0

        while not self.game.is_game_over():
            move_count += 1
            if self.print_move_timers:
                loop_start_time = time.time()
            
            current_player_mcts = self.mcts_white if self.game.board.turn == chess.WHITE else self.mcts_black
            current_board = self.game.board.copy()
            turn_before_move = current_board.turn

            if self.print_move_timers:
                mcts_start_time = time.time()
            
            policy = current_player_mcts.run_search(
                board=current_board,
                num_simulations=self.num_simulations
            )

            if self.print_move_timers:
                mcts_duration = time.time() - mcts_start_time
            
            current_temp = self.temperature if move_count <= self.temp_decay_moves else 0.0
            move_to_play = current_player_mcts.select_move(policy, current_temp)

            if move_to_play is None:
                print(f"[INFO] Move {move_count}: No move could be selected. Ending game early.")
                break

            # --- MODIFIED: Get the value of the next state from MCTS ---
            next_state_value = current_player_mcts.get_next_state_value(move_to_play)
            
            # --- MODIFIED: Store the next_state_value in the history ---
            game_history.append((current_board.fen(), policy, turn_before_move, next_state_value))
            
            self.game.make_move(move_to_play.uci())

            if self.print_move_timers:
                loop_duration = time.time() - loop_start_time
                print(f"[TIMER] Move {move_count}: MCTS search took {mcts_duration:.4f}s. Full loop took {loop_duration:.4f}s.")

        # --- Game is Over ---
        raw_outcome = self.game.get_game_outcome()
        if raw_outcome == 0.0:
            raw_outcome = self.contempt_factor
        
        print(f"\nGame over. Final outcome (White's perspective): {raw_outcome}. Total moves: {len(game_history)}")
        
        # This list will hold the final training data including the new value
        training_data: List[Tuple[str, Dict[chess.Move, float], float, float]] = []
        
        # --- MODIFIED: Unpack the new history tuple and create final training data ---
        for fen_hist, policy_hist, turn_at_state, next_state_value_hist in game_history:
            value_for_state = raw_outcome if turn_at_state == chess.WHITE else -raw_outcome
            training_data.append((fen_hist, policy_hist, value_for_state, next_state_value_hist))

        pgn = None
        try:
            pgn = chess.pgn.Game.from_board(self.game.board)
            pgn.headers["Event"] = "Self-Play Training Game"
            pgn.headers["Site"] = "Juprelle, Wallonia, Belgium"
            pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            pgn.headers["White"] = "MCTS_Agent_v111"
            pgn.headers["Black"] = "MCTS_Agent_v111"
            pgn.headers["Result"] = self.game.board.result(claim_draw=True)
        except Exception as e:
            print(f"[ERROR] Could not generate PGN for the game: {e}")
        
        return training_data, pgn

    def close(self):
        """Closes the Stockfish process."""
        self.game.close()