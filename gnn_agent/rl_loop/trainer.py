import torch
import torch.optim as optim
import torch.nn.functional as F # For policy loss if using raw logits
from torch.nn import CrossEntropyLoss, MSELoss # CrossEntropyLoss might need adjustment for direct policy vector
from typing import Dict, List, Tuple, Any # Added Any
import chess

from ..neural_network.chess_network import ChessNetwork # Corrected relative import
# from ..rl_loop.training_data_manager import TrainingDataManager # Not directly used in Trainer class
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size # ADDED

class Trainer:
    """
    Manages the training process for the ChessNetwork.
    """
    def __init__(self, network: ChessNetwork, learning_rate: float = 0.001, device: torch.device = torch.device("cpu")): # ADDED device
        """
        Initializes the Trainer.

        Args:
            network (ChessNetwork): The neural network model to be trained.
            learning_rate (float): The learning rate for the optimizer.
            device (torch.device): The device to run training on (cpu or cuda).
        """
        self.network = network
        self.device = device # ADDED: Store device
        self.network.to(self.device) # Ensure network is on the correct device
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        # For policy: If network outputs logits and target is dense probability vector,
        # CrossEntropyLoss expects class indices, not a probability distribution.
        # We will use KL Divergence or a direct MSE on probabilities if softmax is applied in the head.
        # Assuming policy_head outputs logits, and MCTS provides a probability distribution.
        # A common loss for policy logits vs MCTS probabilities is Cross-Entropy if MCTS probabilities are one-hot (not usually the case)
        # or KL Divergence if MCTS probabilities are soft.
        # For simplicity, let's treat MCTS policy as target probabilities and use MSE for policy for now,
        # or use CrossEntropy if network outputs logits and target is class indices (not direct probabilities).
        # If policy_head outputs raw logits, and mcts_policy is a probability distribution, a common choice is:
        # F.cross_entropy(pred_policy_logits, target_policy_distribution)
        # self.policy_criterion = CrossEntropyLoss() # This expects target to be class indices
        self.value_criterion = MSELoss()

    def _convert_mcts_policy_to_tensor(self, mcts_policy_dict: Dict[chess.Move, float],
                                       # board_for_context: chess.Board, # May be needed if move_to_index depends on board state
                                       action_space_size: int) -> torch.Tensor:
        """
        Converts an MCTS policy dictionary (move -> prob) to a dense tensor.
        """
        policy_tensor = torch.zeros(action_space_size, device=self.device)
        if not mcts_policy_dict: # Handle cases with no moves (e.g. terminal state where policy might be empty)
            return policy_tensor

        for move, prob in mcts_policy_dict.items():
            try:
                # IMPORTANT: move_to_index should ideally not require the board state here,
                # or the board state corresponding to this policy must be available.
                # Assuming move_to_index is universal or context is handled.
                idx = move_to_index(move, board=None) # Pass None or relevant board if needed
                policy_tensor[idx] = prob
            except IndexError:
                print(f"Warning (Trainer): Move {move.uci()} resulted in an out-of-bounds index from move_to_index during policy tensor creation. Skipping this move.")
            except Exception as e:
                print(f"Warning (Trainer): Error converting move {move.uci()} to index: {e}. Skipping this move.")
        return policy_tensor

    # MODIFIED: train_on_batch to handle GNN input tuples and dict policies
    def train_on_batch(self, batch_data: List[Tuple[Tuple, Dict[chess.Move, float], float]]):
        """
        Performs a single training step on a batch of data.
        The 'board_state' in batch_data is now the GNN input tuple.
        The 'mcts_policy' is a dictionary {chess.Move: probability}.
        """
        if not batch_data:
            return 0.0, 0.0 # Return average losses

        # Unzip the batch data
        # gnn_input_tuples_list: List of (sq_feat, sq_edge, pc_feat, pc_edge, pc_map)
        # mcts_policy_dicts_list: List of Dict[chess.Move, float]
        # game_outcomes_list: List of float
        gnn_input_tuples_list, mcts_policy_dicts_list, game_outcomes_list = zip(*batch_data)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        self.network.train() # Set the network to training mode

        # Process one game state at a time, as the network forward pass is designed for single graphs
        for i in range(len(gnn_input_tuples_list)):
            gnn_input_tuple = gnn_input_tuples_list[i]
            mcts_policy_dict = mcts_policy_dicts_list[i]
            game_outcome = game_outcomes_list[i]

            # Move individual tensors in gnn_input_tuple to device
            processed_gnn_input = tuple(tensor.to(self.device) for tensor in gnn_input_tuple)

            # Convert MCTS policy dictionary to a dense tensor target
            action_space_size = get_action_space_size()
            # We need the board context for move_to_index if it's not universal
            # This is a simplification; ideally, the board state used for mcts_policy_dict
            # should be accessible here to pass to _convert_mcts_policy_to_tensor if needed.
            # For now, assuming move_to_index can work without specific board state or uses a fixed one.
            policy_target_tensor = self._convert_mcts_policy_to_tensor(mcts_policy_dict, action_space_size)
            
            value_target_tensor = torch.tensor([game_outcome], dtype=torch.float32, device=self.device)

            # Zero the gradients for this item
            self.optimizer.zero_grad()

            # Forward pass through the network (expects GNN input tuple unpacked)
            # The ChessNetwork's forward pass adds and removes a batch dim for the heads
            pred_policy_logits, pred_value = self.network(*processed_gnn_input)

            # Calculate loss
            # Policy loss: Cross-entropy between predicted logits and MCTS policy distribution
            # pred_policy_logits is shape [num_actions]
            # policy_target_tensor is shape [num_actions]
            policy_loss = F.cross_entropy(pred_policy_logits.unsqueeze(0), policy_target_tensor.unsqueeze(0))
            
            # Value loss: MSE between predicted value and actual game outcome
            # pred_value is shape [1]
            #value_loss = self.value_criterion(pred_value.unsqueeze(0), value_target_tensor) # Ensure pred_value is also [1] like target
            value_loss = self.value_criterion(pred_value, value_target_tensor)
            
            current_total_loss = policy_loss + value_loss

            # Backward pass and optimization
            current_total_loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        num_samples = len(gnn_input_tuples_list)
        avg_policy_loss = total_policy_loss / num_samples if num_samples > 0 else 0
        avg_value_loss = total_value_loss / num_samples if num_samples > 0 else 0
        
        # print(f"  Batch Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")
        return avg_policy_loss, avg_value_loss