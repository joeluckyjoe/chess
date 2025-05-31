#
# File: policy_value_heads.py
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Policy Head.

        Args:
            x (torch.Tensor): The input tensor from the cross-attention module.
                              Shape: (batch_size, 64, embedding_dim)

        Returns:
            torch.Tensor: The output policy logits.
                          Shape: (batch_size, num_possible_moves)
        """
        # Reshape from (B, 64, D) to (B, D, 8, 8) to treat as a grid
        batch_size, _, embedding_dim = x.shape
        x_grid = x.permute(0, 2, 1).view(batch_size, embedding_dim, 8, 8)
        
        x = F.relu(self.conv1(x_grid))
        
        # Flatten the feature maps
        x = x.view(batch_size, -1)
        
        policy_logits = self.fc1(x)
        
        return policy_logits

class ValueHead(nn.Module):
    """
    The Value Head of the neural network.

    Takes the processed square embeddings and outputs a single scalar value
    estimating the game's outcome from the current player's perspective.
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
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Value Head.

        Args:
            x (torch.Tensor): The input tensor from the cross-attention module.
                              Shape: (batch_size, 64, embedding_dim)

        Returns:
            torch.Tensor: The estimated value of the position, between -1 and 1.
                          Shape: (batch_size, 1)
        """
        # Reshape from (B, 64, D) to (B, D, 8, 8)
        batch_size, _, embedding_dim = x.shape
        x_grid = x.permute(0, 2, 1).view(batch_size, embedding_dim, 8, 8)
        
        x = F.relu(self.conv1(x_grid))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fc1(x))
        
        # Output is a single value, squashed by tanh to be in [-1, 1]
        value = torch.tanh(self.fc2(x))
        
        return value