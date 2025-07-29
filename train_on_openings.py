# FILE: train_on_openings.py
import argparse
import chess
import chess.pgn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import tqdm

# Assuming the project is run from the root directory
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input, CNN_INPUT_CHANNELS
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from hardware_setup import get_device

class OpeningPGNDataset(Dataset):
    def __init__(self, pgn_path, max_moves=20):
        self.pgn_path = pgn_path
        self.max_moves = max_moves
        self.examples = []
        print(f"Loading and processing PGN file: {pgn_path}")
        self._load_games()

    def _load_games(self):
        with open(self.pgn_path, encoding='latin-1') as f:
            game_count = 0
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    game_count += 1
                    board = game.board()
                    for i, move in enumerate(game.mainline_moves()):
                        if i >= self.max_moves:
                            break
                        policy_index = move_to_index(move, board)
                        self.examples.append((board.fen(), policy_index))
                        board.push(move)
                except Exception as e:
                    # Catch potential parsing errors in malformed PGNs
                    # print(f"\nSkipping a game due to parsing error: {e}")
                    continue
        print(f"Processed {len(self.examples)} board positions from {game_count} games.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fen, policy_index = self.examples[idx]
        return fen, policy_index

def load_model(checkpoint_path, device):
    model_params = {
        'gnn_hidden_dim': 128, 'cnn_in_channels': CNN_INPUT_CHANNELS,
        'embed_dim': 256, 'policy_size': get_action_space_size(),
        'gnn_num_heads': 4, 'gnn_metadata': (
            ['square', 'piece'],
            [('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'),
             ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')]
        )
    }
    model = ValueNextStateModel(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    print(f"Model loaded from {checkpoint_path}.")
    return model

def custom_collate_fn(batch):
    device = get_device()
    gnn_data_list = []
    cnn_tensors = []
    policy_indices = []

    for fen, policy_index in batch:
        board = chess.Board(fen)
        gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
        gnn_data_list.append(gnn_data)
        cnn_tensors.append(cnn_tensor)
        policy_indices.append(policy_index)

    gnn_batch = Batch.from_data_list(gnn_data_list)
    cnn_batch = torch.stack(cnn_tensors)
    policy_targets = torch.tensor(policy_indices, dtype=torch.long, device=device)
    
    return gnn_batch, cnn_batch, policy_targets

def main():
    parser = argparse.ArgumentParser(description="Train a model on specific opening lines.")
    parser.add_argument('--pgn-file', type=str, required=True, help="Path to the PGN file with opening games.")
    parser.add_argument('--start-checkpoint', type=str, required=True, help="Path to the pre-trained model to start from.")
    parser.add_argument('--save-path', type=str, required=True, help="Path to save the new opening-specialist model.")
    parser.add_argument('--moves', type=int, default=20, help="Number of half-moves from each game to train on.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=256, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    args = parser.parse_args()

    device = get_device()
    
    dataset = OpeningPGNDataset(args.pgn_file, args.moves)
    # --- FIX: Removed num_workers=2 to prevent the CUDA forking error ---
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    model = load_model(args.start_checkpoint, device)

    print("Freezing Value and Next-State Value heads...")
    for name, param in model.named_parameters():
        if 'value_head' in name:
            param.requires_grad = False
    print("Heads frozen. Only the Policy Head and shared trunk will be trained.")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    policy_criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        total_policy_loss = 0
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for gnn_batch, cnn_batch, policy_targets in progress_bar:
            optimizer.zero_grad()
            
            policy_logits, _, _ = model(gnn_batch, cnn_batch)
            
            loss = policy_criterion(policy_logits, policy_targets)
            
            loss.backward()
            optimizer.step()
            
            total_policy_loss += loss.item()
            progress_bar.set_postfix({'policy_loss': total_policy_loss / (progress_bar.n + 1)})

    torch.save(model.state_dict(), args.save_path)
    print(f"\nTraining complete. Opening specialist model saved to: {args.save_path}")

if __name__ == "__main__":
    main()