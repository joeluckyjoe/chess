import torch
import pytest
from torch_geometric.data import HeteroData, Batch

from gnn_agent.neural_network.hybrid_transformer_model import HybridTransformerModel

@pytest.fixture
def model_config():
    return {
        'gnn_hidden_dim': 64,
        'cnn_in_channels': 14,
        'embed_dim': 128,
        'policy_size': 4672,
        'gnn_num_heads': 2,
        'transformer_nhead': 4,
        'transformer_nlayers': 2,
        'transformer_dim_feedforward': 256,
        'gnn_metadata': (
            ['square', 'piece'],
            [
                ('piece', 'occupies', 'square'),
                ('piece', 'attacks', 'piece'),
                ('piece', 'defends', 'piece'),
                ('square', 'adjacent_to', 'square')
            ]
        )
    }

@pytest.fixture
def sample_sequence_data():
    """Creates a batch of data representing a sequence of board states."""
    seq_len = 10
    piece_feature_dim = 22
    square_feature_dim = 13
    
    gnn_data_list = []
    for _ in range(seq_len):
        data = HeteroData()
        data['piece'].x = torch.randn(32, piece_feature_dim)
        data['square'].x = torch.randn(64, square_feature_dim)
        data['piece', 'occupies', 'square'].edge_index = torch.randint(0, 32, (2, 20))
        data['piece', 'attacks', 'piece'].edge_index = torch.randint(0, 32, (2, 15))
        data['piece', 'defends', 'piece'].edge_index = torch.randint(0, 32, (2, 10))
        data['square', 'adjacent_to', 'square'].edge_index = torch.randint(0, 64, (2, 100))
        gnn_data_list.append(data)

    cnn_tensor = torch.randn(seq_len, 14, 8, 8)
    gnn_batch = Batch.from_data_list(gnn_data_list)
    return gnn_batch, cnn_tensor

def test_model_initialization(model_config):
    """Tests if the model can be initialized without errors."""
    model = HybridTransformerModel(**model_config)
    assert model is not None
    assert isinstance(model.transformer_encoder, torch.nn.TransformerEncoder)

def test_forward_pass_shapes(model_config, sample_sequence_data):
    """Tests the forward pass and checks the output shapes."""
    model = HybridTransformerModel(**model_config)
    gnn_batch, cnn_tensor = sample_sequence_data
    seq_len = cnn_tensor.shape[0]

    policy_logits, value, material = model(gnn_batch, cnn_tensor)

    assert policy_logits.shape == (seq_len, model_config['policy_size'])
    assert value.shape == (seq_len, 1)
    assert material.shape == (seq_len, 1)

def test_backward_pass(model_config, sample_sequence_data):
    """Tests that the backward pass executes and gradients are computed."""
    model = HybridTransformerModel(**model_config)
    gnn_batch, cnn_tensor = sample_sequence_data
    seq_len = cnn_tensor.shape[0]

    policy_logits, value, material = model(gnn_batch, cnn_tensor)

    # Dummy targets
    dummy_policy_target = torch.randint(0, model_config['policy_size'], (seq_len,))
    dummy_value_target = torch.randn(seq_len, 1)
    dummy_material_target = torch.randn(seq_len, 1)

    # Dummy loss calculation
    loss_policy = torch.nn.functional.cross_entropy(policy_logits, dummy_policy_target)
    loss_value = torch.nn.functional.mse_loss(value, dummy_value_target)
    loss_material = torch.nn.functional.mse_loss(material, dummy_material_target)
    total_loss = loss_policy + loss_value + loss_material

    # Check that gradients are initially None
    assert model.policy_head.weight.grad is None

    # Perform backward pass
    total_loss.backward()

    # Check that gradients are now populated
    assert model.policy_head.weight.grad is not None
    
    # MODIFIED: Changed lin_k to lin_l, the correct attribute for GATv2Conv
    edge_type_to_check = ('piece', 'occupies', 'square')
    assert model.gnn.conv1.convs[edge_type_to_check].lin_l.weight.grad is not None
    
    assert model.cnn.conv1.weight.grad is not None
    assert model.transformer_encoder.layers[0].self_attn.out_proj.weight.grad is not None