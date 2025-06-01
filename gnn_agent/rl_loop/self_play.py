import chess
import torch
from typing import List, Tuple, Dict

from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.stockfish_communicator import StockfishCommunicator

class SelfPlay:
    """
    Orchestrates a single game of self-play between two MCTS agents,
    generating training data.
    """
    def __init__(self, mcts_white: MCTS, mcts_black: MCTS, stockfish_path: str, num_simulations: int):
        """
        Initializes a self-play game.

        Args:
            mcts_white: The MCTS search instance for the white player.
            mcts_black: The MCTS search instance for the black player.
            stockfish_path: Path to the Stockfish executable.
        """
        self.mcts_white = mcts_white
        self.mcts_black = mcts_black
        self.game = StockfishCommunicator(stockfish_path)
        self.game.perform_handshake()
        self.training_data: List[Tuple[torch.Tensor, Dict[chess.Move, float], float]] = []
        self.num_simulations = num_simulations

    def play_game(self, num_simulations: int) -> List[Tuple[torch.Tensor, Dict[chess.Move, float], float]]:
        """
        Plays a full game, returning the generated training data.

        Args:
            num_simulations: The number of MCTS simulations to run per move.

        Returns:
            A list of training examples, where each example is a tuple of:
            (board_state_tensor, mcts_policy, game_outcome).
        """
        self.game.reset_board()
        self.training_data = []
        
        # Intermediate storage for states and policies before the outcome is known
        game_history: List[Tuple[torch.Tensor, Dict[chess.Move, float]]] = []

        while not self.game.is_game_over():
            current_player_mcts = self.mcts_white if self.game.board.turn == chess.WHITE else self.mcts_black
            
            # Run the MCTS search to get the best move and the policy
            # The tensor representation is captured inside the MCTS search
            policy, best_move, board_tensor = current_player_mcts.run_search(self.game.board, num_simulations)
            
            if best_move is None:
                # This can happen in a terminal state that MCTS didn't catch, break the loop
                break
                
            game_history.append((board_tensor, policy))
            
            # Play the move
            self.game.make_move(best_move.uci())
            
        # Game is over, determine the outcome
        outcome = self.game.get_game_outcome() # Expected: 1.0, -1.0, or 0.0
        
        # Finalize training data with the determined outcome
        for board_tensor, policy in game_history:
            self.training_data.append((board_tensor, policy, outcome))
            
        return self.training_data