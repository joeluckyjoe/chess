import torch
import os
import re 
from datetime import datetime
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
from typing import Dict, List, Tuple, Any, Optional

# --- Import necessary components ---
import chess
from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size

class Trainer:
    """
    Manages the training process and checkpointing for the ChessNetwork.
    """
    def __init__(self, model_config: Dict[str, Any], network: ChessNetwork = None, learning_rate: float = 0.001, weight_decay: float = 0.0, device: torch.device = torch.device("cpu")):
        """
        Initializes the Trainer. The network can be provided later.
        """
        self.network = network
        self.model_config = model_config
        self.device = device
        self.optimizer = None
        self.value_criterion = MSELoss()
        
        if self.network:
            self.network.to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _initialize_new_network(self) -> Tuple[ChessNetwork, int]:
        """Creates a new network instance based on config and initializes the optimizer."""
        from ..neural_network.gnn_models import SquareGNN, PieceGNN
        from ..neural_network.attention_module import CrossAttentionModule
        from ..neural_network.policy_value_heads import PolicyHead, ValueHead

        print("Creating new network from scratch...")
        # Note: These values should ideally come from the model_config for flexibility
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
        """Extracts the game number from a checkpoint filename."""
        # This regex is made more robust to handle various naming conventions
        match = re.search(r'_game_(\d+)', filepath.name)
        if match:
            return int(match.group(1))
        return -1 # Return -1 if no match, so it won't be chosen as max

    def load_or_initialize_network(self, directory: Path, specific_checkpoint_path: Optional[Path] = None) -> Tuple[ChessNetwork, int]:
        """
        Loads a checkpoint. If a specific path is provided, it uses that.
        Otherwise, it finds the most recent checkpoint in the directory based on game number.
        If no checkpoint is found, it initializes a new network.
        """
        file_to_load = None

        if specific_checkpoint_path:
            if specific_checkpoint_path.exists():
                file_to_load = specific_checkpoint_path
            else:
                print(f"Specified checkpoint not found: {specific_checkpoint_path}. Searching for latest.")
        
        if not file_to_load:
            if not directory.is_dir():
                print(f"Checkpoint directory not found: {directory}. Initializing new network.")
                return self._initialize_new_network()

            files = [f for f in directory.glob('*.pth.tar')]
            if not files:
                print("No checkpoints found. Initializing new network.")
                return self._initialize_new_network()

            try:
                file_to_load = max(files, key=self._get_game_number_from_filename)
            except ValueError:
                print("Could not parse game numbers from any checkpoint files. Initializing new network.")
                return self._initialize_new_network()

        try:
            print(f"Loading checkpoint: {file_to_load}")
            # Initialize network structure first to ensure state_dict can be loaded
            self._initialize_new_network() 

            checkpoint = torch.load(file_to_load, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            game_number = checkpoint.get('game_number', 0)

            # Ensure optimizer state is on the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            print(f"Checkpoint loaded successfully. Resuming from game {game_number + 1}.")
            return self.network, game_number
        except Exception as e:
            print(f"Error loading checkpoint from {file_to_load}: {e}. Initializing new network.")
            return self._initialize_new_network()

    def _convert_mcts_policy_to_tensor(self, mcts_policy_dict: Dict[chess.Move, float],
                                        action_space_size: int) -> torch.Tensor:
        """Converts an MCTS policy dictionary (move -> prob) to a dense tensor."""
        policy_tensor = torch.zeros(action_space_size, device=self.device)
        if not mcts_policy_dict:
            return policy_tensor

        for move, prob in mcts_policy_dict.items():
            try:
                idx = move_to_index(move)
                policy_tensor[idx] = prob
            except Exception as e:
                print(f"Warning (Trainer): Error converting move {move.uci() if move else 'None'} to index: {e}. Skipping.")
        return policy_tensor

    def train_on_batch(self, batch_data: List[Tuple[Any, Dict[chess.Move, float], float]], batch_size: int):
        """Performs a single training step on a batch of data."""
        if not batch_data:
            return 0.0, 0.0

        self.network.train()
        
        # Shuffle the data
        indices = torch.randperm(len(batch_data))
        batch_data = [batch_data[i] for i in indices]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for i in range(0, len(batch_data), batch_size):
            self.optimizer.zero_grad()
            
            batch_chunk = batch_data[i:i+batch_size]
            if not batch_chunk:
                continue
            
            gnn_input_tuples, mcts_policies, game_outcomes = zip(*batch_chunk)

            action_space_size = get_action_space_size()
            policy_targets = torch.stack([self._convert_mcts_policy_to_tensor(p, action_space_size) for p in mcts_policies])
            value_targets = torch.tensor(game_outcomes, dtype=torch.float32, device=self.device).view(-1, 1)
            
            # This part can be slow if not batched properly.
            # A more advanced implementation would batch the gnn_inputs together.
            chunk_policy_loss = 0
            chunk_value_loss = 0

            for j in range(len(batch_chunk)):
                # Assuming gnn_input_tuples[j] is a tuple of tensors
                gnn_input = tuple(tensor.to(self.device) for tensor in gnn_input_tuples[j])
                pred_policy_logits, pred_value = self.network(*gnn_input)

                policy_loss = F.cross_entropy(pred_policy_logits.unsqueeze(0), policy_targets[j].unsqueeze(0))
                value_loss = self.value_criterion(pred_value, value_targets[j])
                
                total_loss = policy_loss + value_loss
                # Normalize loss by chunk size before backward pass
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

