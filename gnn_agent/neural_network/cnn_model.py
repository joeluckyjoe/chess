#
# File: gnn_agent/neural_network/cnn_model.py (Corrected for Hybrid Fusion)
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    A Convolutional Neural Network for processing the chess board.
    
    This version has been modified to output a per-square feature map of shape
    (batch_size, embedding_dim, 8, 8) to be used in a hybrid fusion architecture,
    rather than a single aggregated vector.
    """
    def __init__(self, in_channels: int = 14, embedding_dim: int = 256):
        """
        Initializes the CNN model.

        Args:
            in_channels (int): The number of input channels for the board representation.
            embedding_dim (int): The number of output channels in the final feature map.
        """
        super(CNNModel, self).__init__()
        # A series of convolutional layers that preserve the 8x8 spatial dimension.
        # This is crucial for creating per-square embeddings.
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # Final convolution to produce the desired number of output channels (embedding_dim).
        self.conv3 = nn.Conv2d(256, embedding_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN.

        Args:
            x (torch.Tensor): The input tensor representing the board state.
                              Shape: (batch_size, in_channels, 8, 8)

        Returns:
            torch.Tensor: The output per-square feature map.
                          Shape: (batch_size, embedding_dim, 8, 8)
        """
        # Apply conv layers with GELU activation and batch norm.
        # The spatial dimensions (8x8) are preserved throughout.
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        
        return x
