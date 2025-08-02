import torch
import torch.nn as nn
import chess
from pathlib import Path
from torch_geometric.data import Batch # --- FIX: Added missing import ---

# Add project root to path to allow importing from our modules
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size
from config import config_params
from hardware_setup import get_device

class StyleClassifier:
    """
    An ML-based style classifier that uses a pre-trained feature extractor
    and a trained classification head to score board positions.
    """
    def __init__(self, base_checkpoint_path: str, classifier_head_path: str, device: torch.device):
        self.device = device
        
        # 1. Load the base model (feature extractor)
        base_model = PolicyValueModel(
            gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'],
            cnn_in_channels=14,
            embed_dim=config_params['EMBED_DIM'],
            policy_size=get_action_space_size(),
            gnn_num_heads=config_params['GNN_NUM_HEADS'],
            gnn_metadata=(['square', 'piece'], [('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')])
        ).to(device)
        
        checkpoint = torch.load(base_checkpoint_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        for param in base_model.parameters():
            param.requires_grad = False
        self.base_model = base_model
        
        # 2. Define and load the trained classification head
        self.classifier_head = nn.Sequential(
            nn.Linear(base_model.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        ).to(device)
        
        head_checkpoint = torch.load(classifier_head_path, map_location=device)
        self.classifier_head.load_state_dict(head_checkpoint)
        
        self.base_model.eval()
        self.classifier_head.eval()
        print("ML-based StyleClassifier initialized and models set to eval mode.")

    @torch.no_grad()
    def score_move(self, board: chess.Board, move: chess.Move) -> float:
        """
        Scores a move by evaluating the resulting board state with the ML-based classifier.
        Returns a reward between -0.5 (Tal-like) and +0.5 (Petrosian-like).
        """
        board_after_move = board.copy()
        board_after_move.push(move)
        
        gnn_data, cnn_tensor, _ = convert_to_gnn_input(board_after_move, self.device)
        
        # --- FIX: Use Batch.from_data_list to create a batch of one ---
        gnn_batch = Batch.from_data_list([gnn_data]).to(self.device)
        cnn_tensor = cnn_tensor.unsqueeze(0).to(self.device)

        # --- Perform Inference ---
        batch_size = cnn_tensor.size(0)
        gnn_out = self.base_model.gnn(gnn_batch)
        cnn_out = self.base_model.cnn(cnn_tensor)
        gnn_out_reshaped = gnn_out.view(batch_size, 64, self.base_model.embed_dim).mean(dim=1)
        cnn_out_pooled = cnn_out.view(batch_size, self.base_model.embed_dim, -1).mean(dim=2)
        fused = torch.cat([gnn_out_reshaped, cnn_out_pooled], dim=-1)
        final_embedding = self.base_model.embedding_projection(fused)
        
        style_logit = self.classifier_head(final_embedding)
        
        safety_score = torch.sigmoid(style_logit).item()
        
        reward = safety_score - 0.5
        
        return reward