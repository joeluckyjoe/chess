# gnn_agent/rl_loop/trainer.py

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Dict, List, Tuple
import chess

from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager

class Trainer:
    """
    Manages the training process for the ChessNetwork.
    """
    def __init__(self, network: ChessNetwork, learning_rate: float = 0.001):
        """
        Initializes the Trainer.

        Args:
            network (ChessNetwork): The neural network model to be trained.
            learning_rate (float): The learning rate for the optimizer.
        """
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.policy_loss_fn = CrossEntropyLoss()
        self.value_loss_fn = MSELoss()

    def train_on_batch(self, batch_data: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        """
        Performs a single training step on a batch of data.

        Args:
            batch_data (List[Tuple[torch.Tensor, torch.Tensor, float]]):
                A list of training examples, where each example is a tuple of
                (board_state_tensor, mcts_policy_tensor, game_outcome).
        """
        if not batch_data:
            return None, None

        # Unzip the batch data
        board_states, mcts_policies, game_outcomes = zip(*batch_data)

        # Stack the tensors
        board_states_tensor = torch.stack(board_states)
        mcts_policies_tensor = torch.stack(mcts_policies)
        game_outcomes_tensor = torch.tensor(game_outcomes, dtype=torch.float32).view(-1, 1)

        # Zero the gradients
        self.optimizer.zero_grad()

        # Forward pass
        pred_policies, pred_values = self.network(board_states_tensor)

        # Calculate loss
        policy_loss = self.policy_loss_fn(pred_policies, mcts_policies_tensor)
        value_loss = self.value_loss_fn(pred_values, game_outcomes_tensor)
        total_loss = policy_loss + value_loss

        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()