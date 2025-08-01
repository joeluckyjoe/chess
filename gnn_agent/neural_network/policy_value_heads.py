#
# File: gnn_agent/neural_network/policy_value_heads.py (Updated for Phase BI)
#
"""
This file defines the Policy and Value Head modules for the MCTS RL Chess Agent.
These modules take the final processed embeddings from the core GNN/attention
network and produce the policy (move probabilities) and value (estimated game
outcome) respectively.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PolicyHead(nn.Module):
    """
    The Policy Head of the neural network.

    Takes the processed square embeddings and outputs a probability distribution
    over all possible moves.
    """
    def __init__(self, embedding_dim: int, num_possible_moves: int = 4672):
        """
        Initializes the PolicyHead.

        Args:
            embedding_dim (int): The dimension of the input embeddings for each square.
            num_possible_moves (int): The size of the action space.
                                      Default is 4672 for chess.
        """
        super(PolicyHead, self).__init__()
        # Use a conv layer to create policy-specific feature maps.
        # It acts like a per-square linear layer initially.
        self.conv1 = nn.Conv2d(in_channels=embedding_dim, out_channels=2, kernel_size=1)
        
        # The output of the conv layer will be (B, 2, 8, 8), which flattens to B, 128
        self.fc1 = nn.Linear(2 * 8 * 8, num_possible_moves)

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Policy Head.

        Args:
            x (torch.Tensor): The input tensor from the core network.
                              Shape: (total_squares_in_batch, embedding_dim)
            batch (torch.Tensor): The batch tensor indicating which graph each square belongs to.
                                  Shape: (total_squares_in_batch,)

        Returns:
            torch.Tensor: The output policy logits.
                          Shape: (batch_size, num_possible_moves)
        """
        if batch is None:
            batch_size = 1
        else:
            batch_size = batch.max().item() + 1
        
        embedding_dim = x.size(1)

        # Reshape from (B * 64, D) to (B, D, 8, 8)
        x_grid = x.view(batch_size, 64, embedding_dim)
        x_grid = x_grid.permute(0, 2, 1).view(batch_size, embedding_dim, 8, 8)
        
        x = F.gelu(self.conv1(x_grid))
        
        # Flatten the feature maps. Use .reshape() instead of .view() to handle
        # potentially non-contiguous tensors after the convolution.
        x = x.reshape(batch_size, -1)
        
        policy_logits = self.fc1(x)
        
        return policy_logits

class ValueHead(nn.Module):
    """
    The Value Head of the neural network. (Updated for Phase BI)

    Takes the processed square embeddings and outputs two separate scalar values:
    1.  The estimated game's outcome from the current player's perspective.
    2.  The estimated material balance from the current player's perspective.
    """
    def __init__(self, embedding_dim: int):
        """
        Initializes the ValueHead.

        Args:
            embedding_dim (int): The dimension of the input embeddings for each square.
        """
        super(ValueHead, self).__init__()
        # Summarize spatial info
        self.conv1 = nn.Conv2d(in_channels=embedding_dim, out_channels=1, kernel_size=1)
        
        # The output of the conv layer will be (B, 1, 8, 8), which flattens to B, 64
        self.fc1 = nn.Linear(1 * 8 * 8, 256)

        # --- MODIFICATION FOR PHASE BI ---
        # Create two separate final layers for the two outputs
        self.value_fc = nn.Linear(256, 1)
        self.material_fc = nn.Linear(256, 1)
        # --- END MODIFICATION ---

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Value Head.

        Args:
            x (torch.Tensor): The input tensor from the core network.
                              Shape: (total_squares_in_batch, embedding_dim)
            batch (torch.Tensor): The batch tensor indicating which graph each square belongs to.
                                  Shape: (total_squares_in_batch,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - value (torch.Tensor): Estimated game outcome, in [-1, 1]. Shape: (batch_size, 1)
                - material_balance (torch.Tensor): Estimated material balance. Shape: (batch_size, 1)
        """
        if batch is None:
            batch_size = 1
        else:
            batch_size = batch.max().item() + 1

        embedding_dim = x.size(1)

        # Reshape from (B * 64, D) to (B, D, 8, 8)
        x_grid = x.view(batch_size, 64, embedding_dim)
        x_grid = x_grid.permute(0, 2, 1).view(batch_size, embedding_dim, 8, 8)
        
        x = F.gelu(self.conv1(x_grid))
        
        # Flatten. Use .reshape() instead of .view() to handle
        # potentially non-contiguous tensors after the convolution.
        x = x.reshape(batch_size, -1)
        
        x = F.gelu(self.fc1(x))
        
        # --- MODIFICATION FOR PHASE BI ---
        # Calculate the two separate outputs
        # 1. Game outcome value, squashed by tanh to be in [-1, 1]
        value = torch.tanh(self.value_fc(x))
        
        # 2. Material balance, returned as a raw logit for regression
        material_balance = self.material_fc(x)
        # --- END MODIFICATION ---
        
        return value, material_balance
