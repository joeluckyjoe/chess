import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    A simple Convolutional Neural Network for processing the chess board.
    It takes a multi-channel 8x8 board representation and outputs a
    flattened feature vector.
    """
    def __init__(self, in_channels: int = 14, embedding_dim: int = 256):
        """
        Initializes the CNN model.

        Args:
            in_channels (int): The number of input channels for the board
                               representation (e.g., 12 for piece types + 2 for colors).
            embedding_dim (int): The size of the output feature vector.
        """
        super(CNNModel, self).__init__()
        # Conv Layer 1: preserves spatial dimensions (8x8)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # Conv Layer 2: preserves spatial dimensions (8x8)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Conv Layer 3: downsamples to 4x4
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        
        # Calculate the flattened size after convolutions
        # Input: (B, C, 8, 8) -> conv1 -> (B, 64, 8, 8) -> conv2 -> (B, 128, 8, 8)
        # -> conv3 -> (B, 128, 4, 4)
        flattened_size = 128 * 4 * 4
        
        # Fully connected layers to produce the final embedding
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN.

        Args:
            x (torch.Tensor): The input tensor representing the board state.
                              Shape: (batch_size, in_channels, 8, 8)

        Returns:
            torch.Tensor: The output embedding vector.
                          Shape: (batch_size, embedding_dim)
        """
        # Apply conv layers with GELU activation and batch norm
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Apply fully connected layers with GELU activation
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x