import torch
import os
import re
from datetime import datetime
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
from typing import Dict, List, Tuple, Any, Optional

import chess
from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input

class Trainer:
    """
    Manages the training process and checkpointing for the ChessNetwork.
    """
    def __init__(self, model_config: Dict[str, Any], network: ChessNetwork = None, learning_rate: float = 0.001, weight_decay: float = 0.0, device: torch.device = torch.device("cpu")):
        self.network = network
        self.model_config = model_config
        self.device = device
        self.optimizer = None
        self.value_criterion = MSELoss()
        
        if self.network:
            self.network.to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _initialize_new_network(self) -> Tuple[ChessNetwork, int]:
        from ..neural_network.gnn_models import SquareGNN, PieceGNN
        from ..neural_network.attention_module import CrossAttentionModule
        from ..neural_network.policy_value_heads import PolicyHead, ValueHead

        print("Creating new network from scratch...")
        square_gnn = SquareGNN(in_features=12, hidden_features=256, out_features=128, heads=4)
        piece_gnn = PieceGNN(in_channels=12, hidden_channels=256, out_channels=128)
        cross_attention = CrossAttentionModule(sq_embed_dim=128, pc_embed_dim=128, num_heads=4)
        policy_head = PolicyHead(embedding_dim=128, num_possible_moves=get_action_space_size())
        value_head = ValueHead(embedding_dim=128)

        self.network = ChessNetwork(
            square_gnn=square_gnn,
            piece_gnn=piece_gnn,
            cross_attention=cross_attention,
            policy_head=policy_head,
            value_head=value_head
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.model_config['LEARNING_RATE'], 
            weight_decay=self.model_config['WEIGHT_DECAY']
        )
        return self.network, 0

    def _get_game_number_from_filename(self, filepath: Path) -> int:
        match = re.search(r'_game_(\d+)', filepath.name)
        if match:
            return int(match.group(1))
        return -1

    def load_or_initialize_network(self, directory: Path, specific_checkpoint_path: Optional[Path] = None) -> Tuple[ChessNetwork, int]:
        file_to_load = None
        if specific_checkpoint_path and specific_checkpoint_path.exists():
            file_to_load = specific_checkpoint_path
        elif not specific_checkpoint_path:
            if directory.is_dir():
                files = [f for f in directory.glob('*.pth.tar')]
                if files:
                    file_to_load = max(files, key=self._get_game_number_from_filename)

        if not file_to_load:
            print("No checkpoint found or specified. Initializing new network.")
            return self._initialize_new_network()

        try:
            print(f"Loading checkpoint: {file_to_load}")
            self._initialize_new_network()
            checkpoint = torch.load(file_to_load, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            game_number = checkpoint.get('game_number', 0)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print(f"Checkpoint loaded successfully. Resuming from game {game_number + 1}.")
            return self.network, game_number
        except Exception as e:
            print(f"Error loading checkpoint {file_to_load}: {e}. Initializing new network.")
            return self._initialize_new_network()

    def _convert_mcts_policy_to_tensor(self, mcts_policy_dict: Dict[chess.Move, float], board: chess.Board, action_space_size: int) -> torch.Tensor:
        """Converts an MCTS policy dictionary to a dense tensor, now with board context."""
        policy_tensor = torch.zeros(action_space_size, device=self.device)
        if not mcts_policy_dict:
            return policy_tensor
        for move, prob in mcts_policy_dict.items():
            try:
                # --- BUG FIX: Pass the board object to move_to_index ---
                idx = move_to_index(move, board)
                policy_tensor[idx] = prob
            except Exception as e:
                # The print statement below was removed as it was flooding the logs. The try/except is sufficient.
                # print(f"Warning (Trainer): Error converting move {move.uci() if move else 'None'} to index: {e}. Skipping.")
                pass
        return policy_tensor

    def train_on_batch(self, batch_data: List[Tuple[str, Dict[chess.Move, float], float]], batch_size: int):
        """
        Performs a single training step on a batch of data.
        The data format is now (FEN, policy_dict, outcome).
        """
        if not batch_data:
            return 0.0, 0.0

        self.network.train()
        
        indices = torch.randperm(len(batch_data))
        batch_data = [batch_data[i] for i in indices]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for i in range(0, len(batch_data), batch_size):
            self.optimizer.zero_grad()
            
            batch_chunk = batch_data[i:i+batch_size]
            if not batch_chunk:
                continue
            
            # --- BUG FIX: Unzip the new data format ---
            fen_strings, mcts_policies, game_outcomes = zip(*batch_chunk)

            # --- BUG FIX: Process FENs into boards and tensors inside the loop ---
            boards = [chess.Board(fen) for fen in fen_strings]
            gnn_inputs = [convert_to_gnn_input(b, device=self.device) for b in boards]

            action_space_size = get_action_space_size()
            policy_targets = torch.stack([
                self._convert_mcts_policy_to_tensor(p, b, action_space_size) 
                for p, b in zip(mcts_policies, boards)
            ])
            value_targets = torch.tensor(game_outcomes, dtype=torch.float32, device=self.device).view(-1, 1)
            
            # Batching graph data is complex; we process one by one for clarity and safety.
            chunk_policy_loss = 0
            chunk_value_loss = 0
            for j in range(len(batch_chunk)):
                pred_policy_logits, pred_value = self.network(*gnn_inputs[j])
                policy_loss = F.cross_entropy(pred_policy_logits.unsqueeze(0), policy_targets[j].unsqueeze(0))
                value_loss = self.value_criterion(pred_value, value_targets[j])
                
                total_loss = policy_loss + value_loss
                (total_loss / len(batch_chunk)).backward()

                chunk_policy_loss += policy_loss.item()
                chunk_value_loss += value_loss.item()
            
            self.optimizer.step()
            total_policy_loss += chunk_policy_loss
            total_value_loss += chunk_value_loss

        num_samples = len(batch_data)
        avg_policy_loss = total_policy_loss / num_samples if num_samples > 0 else 0
        avg_value_loss = total_value_loss / num_samples if num_samples > 0 else 0
        
        return avg_policy_loss, avg_value_loss
    
    def save_checkpoint(self, directory: Path, game_number: int, filename_override: Optional[str] = None):
        """Saves the model and optimizer state. Can use a specific filename."""
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)

        if filename_override:
            filename = filename_override
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_game_{game_number}_{timestamp}.pth.tar"
        
        filepath = directory / filename
        state = {
            'game_number': game_number,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config_params': self.model_config,
        }
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")
