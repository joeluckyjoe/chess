import chess
import torch
import random
from typing import List, Tuple, Dict, Any
import chess.pgn
from datetime import datetime

from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

class MentorPlay:
    """
    Orchestrates a single game between an MCTS agent and a Stockfish mentor,
    generating training data and a PGN of the game.
    """
    def __init__(self, mcts_agent: MCTS, stockfish_path: str, stockfish_depth: int, num_simulations: int, agent_color_str: str = "random"):
        """
        Initializes a mentor game player.
        """
        self.mcts_agent = mcts_agent
        self.stockfish_depth = stockfish_depth
        self.num_simulations = num_simulations
        # --- MODIFIED: Store the configured color string, don't determine the color yet ---
        self.agent_color_config = agent_color_str
        
        print("Initializing Stockfish for MentorPlay...")
        self.stockfish_player = StockfishCommunicator(stockfish_path)
        self.stockfish_player.perform_handshake()

    def play_game(self) -> Tuple[List[Tuple[Any, Dict[chess.Move, float], float]], chess.pgn.Game]:
        """
        Plays a full game, returning training data and the PGN object.
        """
        # --- MODIFIED: Determine agent color at the start of each game ---
        if self.agent_color_config == "random":
            agent_color = random.choice([chess.WHITE, chess.BLACK])
        elif self.agent_color_config == "white":
            agent_color = chess.WHITE
        else: # black
            agent_color = chess.BLACK
        
        print(f"--- Starting a new mentor game: Agent plays as {chess.COLOR_NAMES[agent_color]} ---")
        
        self.stockfish_player.reset_board()
        board = self.stockfish_player.board

        game_history = [] 
        
        while not self.stockfish_player.is_game_over():
            if board.turn == agent_color:
                # --- Agent's Turn ---
                print("Agent's turn...")
                policy, best_move, board_tensor = self.mcts_agent.run_search(
                    board.copy(),
                    self.num_simulations
                )
                
                game_history.append((board_tensor, policy, agent_color)) 
                
                if not best_move:
                    print("MCTS returned no move, ending game.")
                    break
                
                move_uci = best_move.uci()
                # print(f"Agent plays: {move_uci}") # This is redundant with the MCTS output
                self.stockfish_player.make_move(move_uci)

            else: # Stockfish's turn
                print("Mentor's turn...")
                move_uci = self.stockfish_player.get_best_move(self.stockfish_depth)
                # print(f"Mentor plays: {move_uci}") # This is redundant with Stockfish output
                self.stockfish_player.make_move(move_uci)

        # --- Game Over ---
        raw_outcome = self.stockfish_player.get_game_outcome()
        print(f"Mentor game over. Raw outcome (White's perspective): {raw_outcome}")
        
        # The logic for calculating agent_perspective_result is now simpler
        # because the raw_outcome (1.0 for white win, -1.0 for black win)
        # aligns with the values we need.
        agent_perspective_result = 0.0
        if raw_outcome is not None:
            if agent_color == chess.WHITE:
                agent_perspective_result = raw_outcome
            else: # Agent is black
                agent_perspective_result = -raw_outcome

        training_data = []
        for board_tensor_hist, policy_hist, _ in game_history:
            training_data.append((board_tensor_hist, policy_hist, agent_perspective_result))

        # --- Generate PGN ---
        pgn = None
        try:
            pgn = chess.pgn.Game()
            pgn.headers["Event"] = "Mentor Training Game"
            pgn.headers["Site"] = "Colab Environment"
            pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            pgn.headers["White"] = "MCTS_Agent" if agent_color == chess.WHITE else "Stockfish_Mentor"
            pgn.headers["Black"] = "MCTS_Agent" if agent_color == chess.BLACK else "Stockfish_Mentor"
            pgn.headers["Result"] = self.stockfish_player.board.result(claim_draw=True)
            
            if self.stockfish_player.board.move_stack:
                node = pgn.add_main_variation(self.stockfish_player.board.move_stack[0])
                for i in range(1, len(self.stockfish_player.board.move_stack)):
                    node = node.add_main_variation(self.stockfish_player.board.move_stack[i])
        except Exception as e:
            print(f"[ERROR] Could not generate PGN for the game: {e}")

        return training_data, pgn

    def close(self):
        """Closes the Stockfish process."""
        self.stockfish_player.close()
