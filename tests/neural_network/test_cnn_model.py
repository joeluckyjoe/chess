import pytest
import torch
from gnn_agent.neural_network.cnn_model import CNNModel

@pytest.fixture
def cnn_config():
    """Provides a standard configuration for the CNN model."""
    return {
        "in_channels": 14,
        "embedding_dim": 256
    }

def test_cnn_model_instantiation(cnn_config):
    """Tests if the CNNModel can be instantiated correctly."""
    model = CNNModel(in_channels=cnn_config["in_channels"], 
                     embedding_dim=cnn_config["embedding_dim"])
    assert model is not None, "Model instantiation failed."

def test_cnn_model_forward_pass_single_input(cnn_config):
    """
    Tests the forward pass with a single input tensor.
    """
    model = CNNModel(in_channels=cnn_config["in_channels"], 
                     embedding_dim=cnn_config["embedding_dim"])
    
    # Create a dummy input tensor for a single board
    # Shape: (batch_size=1, in_channels, height, width)
    single_input = torch.randn(1, cnn_config["in_channels"], 8, 8)
    
    output = model(single_input)
    
    # Check that the output has the correct shape
    expected_shape = (1, cnn_config["embedding_dim"])
    assert output.shape == expected_shape, \
        f"Output shape for single input is incorrect. Expected {expected_shape}, got {output.shape}"

def test_cnn_model_forward_pass_batch_input(cnn_config):
    """
    Tests the forward pass with a batch of input tensors.
    """
    model = CNNModel(in_channels=cnn_config["in_channels"], 
                     embedding_dim=cnn_config["embedding_dim"])
    
    batch_size = 16
    # Create a dummy input tensor for a batch of boards
    # Shape: (batch_size, in_channels, height, width)
    batch_input = torch.randn(batch_size, cnn_config["in_channels"], 8, 8)
    
    output = model(batch_input)
    
    # Check that the output has the correct shape for the batch
    expected_shape = (batch_size, cnn_config["embedding_dim"])
    assert output.shape == expected_shape, \
        f"Output shape for batch input is incorrect. Expected {expected_shape}, got {output.shape}"

def test_cnn_model_output_is_not_nan(cnn_config):
    """
    Tests that the model output does not contain NaNs.
    """
    model = CNNModel(in_channels=cnn_config["in_channels"], 
                     embedding_dim=cnn_config["embedding_dim"])
    
    batch_input = torch.randn(4, cnn_config["in_channels"], 8, 8)
    output = model(batch_input)
    
    assert not torch.isnan(output).any(), "Model output contains NaNs."