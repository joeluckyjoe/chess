# FILENAME: gnn_agent/rl_loop/self_play.py
import chess
import torch
from typing import List, Tuple, Dict
import time
import chess.pgn
from datetime import datetime
import random
from stockfish import Stockfish

from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator
# The problematic board_converter import has been removed.

class SelfPlay:
    """
    Orchestrates a single game of self-play for the ValueNextStateModel,
    with an option for hybrid mentor-corrected play.
    """
    def __init__(self, network: ValueNextStateModel, device: torch.device,
                 mcts_white: MCTS, mcts_black: MCTS, 
                 stockfish_path: str, num_simulations: int,
                 mentor_engine: Stockfish,
                 mentor_intervention_prob: float = 0.0,
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
        self.mentor_engine = mentor_engine
        self.mentor_intervention_prob = mentor_intervention_prob
        self.interventions_made = 0

    def play_game(self) -> Tuple[List[Tuple[str, Dict[chess.Move, float], float, float]], chess.pgn.Game]:
        """
        Plays a full game, generating training data.
        If mentor_intervention_prob > 0, it will periodically force the mentor's move.
        Returns tuples of (FEN, policy, value, next_state_value).
        """
        if self.mentor_intervention_prob > 0:
            print(f"Starting a new hybrid self-play game (Intervention Prob: {self.mentor_intervention_prob:.2f})...")
        else:
            print("Starting a new self-play game...")
            
        self.game.reset_board()
        self.interventions_made = 0
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = "Hybrid Self-Play Training Game" if self.mentor_intervention_prob > 0 else "Self-Play Training Game"
        pgn_game.headers["Site"] = "Juprelle, Wallonia, Belgium"
        pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        pgn_game.headers["White"] = "Agent_MCTS"
        pgn_game.headers["Black"] = "Agent_MCTS"
        if self.mentor_intervention_prob > 0:
            pgn_game.headers["InterventionRate"] = f"{self.mentor_intervention_prob:.2f}"
        
        pgn_node = pgn_game

        game_history: List[Tuple[str, Dict[chess.Move, float], bool, float]] = []
        move_count = 0

        while not self.game.is_game_over():
            move_count += 1
            current_board = self.game.board.copy()
            turn_before_move = current_board.turn
            
            move_to_play = None
            policy = {}
            is_intervention = False

            # --- PHASE BQ: HYBRID MENTOR-RL LOGIC ---
            if random.random() < self.mentor_intervention_prob:
                is_intervention = True
                self.interventions_made += 1
                self.mentor_engine.set_fen_position(current_board.fen())
                best_move_uci = self.mentor_engine.get_best_move_time(100) # 100ms
                if best_move_uci:
                    move_to_play = chess.Move.from_uci(best_move_uci)
                    # Create a "one-hot" policy targeting the mentor's move
                    for legal_move in current_board.legal_moves:
                        policy[legal_move] = 1.0 if legal_move == move_to_play else 0.0
                    print(f"** Mentor Intervention on move {move_count}: Chose {move_to_play.uci()} **")
                else:
                    is_intervention = False # Fallback if mentor fails

            # --- STANDARD MCTS-DRIVEN MOVE ---
            if not is_intervention:
                current_player_mcts = self.mcts_white if turn_before_move == chess.WHITE else self.mcts_black
                policy = current_player_mcts.run_search(board=current_board, num_simulations=self.num_simulations)
                current_temp = self.temperature if move_count <= self.temp_decay_moves else 0.0
                move_to_play = current_player_mcts.select_move(policy, current_temp)

            if move_to_play is None or move_to_play not in current_board.legal_moves:
                print(f"[INFO] Move {move_count}: No valid move could be selected. Ending game.")
                break
            
            # --- MODIFIED: Use the new MCTS method to get the next state value ---
            temp_board = current_board.copy()
            temp_board.push(move_to_play)
            
            # Evaluate the board *after* the move. Negate it to get the value from the
            # perspective of the player *making* the move.
            next_state_value = -self.mcts_white.evaluate_single_board(temp_board)
            
            # Store the data for this state
            game_history.append((current_board.fen(), policy, turn_before_move, next_state_value))
            
            # Make the move on the board and in the PGN
            self.game.make_move(move_to_play.uci())
            pgn_node = pgn_node.add_variation(move_to_play)
            if is_intervention:
                pgn_node.comment = "Mentor Intervention"

        # --- Game is Over ---
        raw_outcome = self.game.get_game_outcome()
        if raw_outcome == 0.0:
            raw_outcome = self.contempt_factor
        
        print(f"\nGame over. Final outcome (White's perspective): {raw_outcome}. Total moves: {move_count}. Interventions: {self.interventions_made}.")
        
        training_data: List[Tuple[str, Dict[chess.Move, float], float, float]] = []
        for fen_hist, policy_hist, turn_at_state, next_state_value_hist in game_history:
            value_for_state = raw_outcome if turn_at_state == chess.WHITE else -raw_outcome
            training_data.append((fen_hist, policy_hist, value_for_state, next_state_value_hist))

        pgn_game.headers["Result"] = self.game.board.result(claim_draw=True)
        return training_data, pgn_game

    def close(self):
        """Closes the local Stockfish communicator process."""
        self.game.close()