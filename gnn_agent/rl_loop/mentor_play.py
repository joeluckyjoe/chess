import chess
import torch
import random
from typing import List, Tuple, Dict, Any

from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

class MentorPlay:
    """
    Orchestrates a single game between an MCTS agent and a Stockfish mentor,
    generating training data from the agent's perspective.
    """
    # --- FIXED: Added num_simulations to the constructor ---
    def __init__(self, mcts_agent: MCTS, stockfish_path: str, stockfish_depth: int, num_simulations: int, agent_color_str: str = "random"):
        """
        Initializes a mentor game.

        Args:
            mcts_agent: The MCTS search instance for our agent.
            stockfish_path: Path to the Stockfish executable.
            stockfish_depth: The search depth for Stockfish.
            num_simulations: The number of MCTS simulations to run per agent move.
            agent_color_str: The color our agent plays ("white", "black", or "random").
        """
        self.mcts_agent = mcts_agent
        self.stockfish_depth = stockfish_depth
        self.num_simulations = num_simulations # <-- STORE THE VALUE
        
        print("Initializing Stockfish for MentorPlay...")
        self.stockfish_player = StockfishCommunicator(stockfish_path)
        self.stockfish_player.perform_handshake()

        if agent_color_str == "random":
            self.agent_color = random.choice([chess.WHITE, chess.BLACK])
        elif agent_color_str == "white":
            self.agent_color = chess.WHITE
        else:
            self.agent_color = chess.BLACK

        print(f"Mentor game setup: Agent plays as {chess.COLOR_NAMES[self.agent_color]}")

    def play_game(self) -> List[Tuple[Any, Dict[chess.Move, float], float]]:
        """
        Plays a full game, returning training data from the agent's perspective.
        """
        print("Starting a new mentor game...")
        self.stockfish_player.reset_board()
        board = self.stockfish_player.board

        game_history = [] # Stores (board_tensor, policy, turn_at_state)
        
        while not self.stockfish_player.is_game_over():
            if board.turn == self.agent_color:
                # --- Agent's Turn ---
                # --- FIXED: Use self.num_simulations ---
                policy, best_move, board_tensor = self.mcts_agent.run_search(
                    board.copy(),
                    self.num_simulations
                )
                
                game_history.append((board_tensor, policy, self.agent_color)) 
                
                if not best_move:
                    print("MCTS returned no move, ending game.")
                    break
                
                move_uci = best_move.uci()
                print(f"Agent plays: {move_uci}")
                self.stockfish_player.make_move(move_uci)

            else: # Stockfish's turn
                print("Mentor's turn...")
                move_uci = self.stockfish_player.get_best_move(self.stockfish_depth)
                print(f"Mentor plays: {move_uci}")
                self.stockfish_player.make_move(move_uci)

        # --- Game Over ---
        raw_outcome = self.stockfish_player.get_game_outcome()
        print(f"Mentor game over. Raw outcome (White's perspective): {raw_outcome}")
        
        agent_perspective_result = 0.0
        if raw_outcome is not None:
            if self.agent_color == chess.WHITE:
                agent_perspective_result = raw_outcome
            else:
                agent_perspective_result = -raw_outcome

        training_data = []
        for board_tensor_hist, policy_hist, _ in game_history:
            training_data.append((board_tensor_hist, policy_hist, agent_perspective_result))

        return training_data

    def close(self):
        """Closes the Stockfish process."""
        self.stockfish_player.close()
