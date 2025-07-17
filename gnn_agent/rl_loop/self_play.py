import chess
import torch
from typing import List, Tuple, Dict, Any
import time
import chess.pgn
from datetime import datetime
import random

# Import the correct network and MCTS classes
from gnn_agent.neural_network.hybrid_rnn_model import HybridRNNModel
from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

class SelfPlay:
    """
    Orchestrates a single game of self-play, updated for a stateful RNN model.
    """
    def __init__(self, network: HybridRNNModel, device: torch.device, 
                 mcts_white: MCTS, mcts_black: MCTS, 
                 stockfish_path: str, num_simulations: int, 
                 temperature: float = 1.0, temp_decay_moves: int = 30, 
                 print_move_timers: bool = False, contempt_factor: float = 0.0):
        """
        MODIFIED FOR PHASE BJ: Added `network` and `device` to manage the RNN state.
        """
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

    def play_game(self) -> Tuple[List[Tuple[str, Dict[chess.Move, float], float]], chess.pgn.Game]:
        """
        Plays a full game, managing the RNN hidden state across moves.
        """
        print("Starting a new self-play game...")
        self.game.reset_board()

        game_history: List[Tuple[str, Dict[chess.Move, float], bool]] = []
        training_data: List[Tuple[str, Dict[chess.Move, float], float]] = []
        move_count = 0

        # --- PHASE BJ MODIFICATION: Initialize Hidden State ---
        num_layers = self.network.num_rnn_layers
        hidden_dim = self.network.rnn_hidden_dim
        # The hidden state has a "batch size" of 1, as we play one game at a time.
        hidden_state = torch.zeros((num_layers, 1, hidden_dim), device=self.device)
        # --- END MODIFICATION ---

        while not self.game.is_game_over():
            move_count += 1
            if self.print_move_timers:
                loop_start_time = time.time()

            current_player_mcts = self.mcts_white if self.game.board.turn == chess.WHITE else self.mcts_black
            current_board = self.game.board.copy()
            turn_before_move = current_board.turn

            if self.print_move_timers:
                mcts_start_time = time.time()
            
            # --- PHASE BJ MODIFICATION: Pass and receive the hidden state ---
            policy, new_hidden_state = current_player_mcts.run_search(
                board=current_board,
                num_simulations=self.num_simulations,
                hidden_state=hidden_state
            )
            # Update the hidden state for the next turn
            hidden_state = new_hidden_state
            # --- END MODIFICATION ---

            if self.print_move_timers:
                mcts_duration = time.time() - mcts_start_time
            
            current_temp = self.temperature if move_count <= self.temp_decay_moves else 0.0
            move_to_play = current_player_mcts.select_move(policy, current_temp)

            if move_to_play is None:
                print(f"[INFO] Move {move_count}: No move could be selected. Ending game early.")
                break

            game_history.append((current_board.fen(), policy, turn_before_move))
            self.game.make_move(move_to_play.uci())

            if self.print_move_timers:
                loop_duration = time.time() - loop_start_time
                print(f"[TIMER] Move {move_count}: MCTS search took {mcts_duration:.4f}s. Full loop took {loop_duration:.4f}s.")

        # --- Game is Over --- (No changes below this line)
        raw_outcome = self.game.get_game_outcome()
        if raw_outcome == 0.0:
            raw_outcome = self.contempt_factor
        
        print(f"\nGame over. Final outcome (White's perspective): {raw_outcome}. Total moves: {len(game_history)}")
        
        for fen_hist, policy_hist, turn_at_state in game_history:
            value_for_state = 0.0
            if raw_outcome is not None:
                if turn_at_state == chess.WHITE:
                    value_for_state = raw_outcome
                else:
                    value_for_state = -raw_outcome
            training_data.append((fen_hist, policy_hist, value_for_state))

        pgn = None
        try:
            pgn = chess.pgn.Game.from_board(self.game.board)
            pgn.headers["Event"] = "Self-Play Training Game"
            pgn.headers["Site"] = "Herstal, Wallonia, Belgium"
            pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            pgn.headers["White"] = "MCTS_Agent_v107"
            pgn.headers["Black"] = "MCTS_Agent_v107"
        except Exception as e:
            print(f"[ERROR] Could not generate PGN for the game: {e}")
        
        return training_data, pgn

    def close(self):
        """Closes the Stockfish process."""
        self.game.close()