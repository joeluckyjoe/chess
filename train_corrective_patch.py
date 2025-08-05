import torch
import torch.optim as optim
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import random
from torch_geometric.data import Batch

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))

from config import get_paths, config_params
from gnn_agent.neural_network.policy_value_model import PolicyValueModel as EncoderPolicyValueModel
from gnn_agent.neural_network.temporal_model import TemporalPolicyValueModel
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size
from hardware_setup import get_device

GNN_METADATA = (['square', 'piece'],[('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')])

# --- Configuration ---
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

# <<< ADDED: A dedicated batch preparation function for this script.
def prepare_corrective_batch(batch_data, device):
    """
    Prepares a batch for corrective training. Unlike the main training
    function, this does not process value targets.
    """
    all_gnn_data = []
    cnn_sequences = []

    for item in batch_data:
        gnn_sequence_for_step = [s[0] for s in item['state_sequence']]
        cnn_sequence_for_step = [s[1] for s in item['state_sequence']]
        all_gnn_data.extend(gnn_sequence_for_step)
        cnn_sequences.append(torch.stack(cnn_sequence_for_step))

    gnn_batch = Batch.from_data_list(all_gnn_data).to(device)
    cnn_batch = torch.stack(cnn_sequences).to(device)
    target_policies = torch.stack([item['policy'] for item in batch_data]).to(device)

    return gnn_batch, cnn_batch, target_policies


def main():
    parser = argparse.ArgumentParser(description="Perform a surgical patch on a model using a corrective dataset.")
    parser.add_argument('--encoder_checkpoint', type=str, required=True, help="Path to the original PRE-TRAINED ENCODER model from Phase C.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the TEMPORAL model checkpoint to be patched.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the corrective_dataset.pt file.")
    args = parser.parse_args()

    paths = get_paths()
    device = get_device()
    print(f"Using device: {device}")

    # --- 1. Load the Model to be Patched ---
    print(f"Loading base encoder from: {args.encoder_checkpoint}")
    encoder = EncoderPolicyValueModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
        embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
    ).to(device)
    encoder_checkpoint = torch.load(args.encoder_checkpoint, map_location=device)
    encoder_state_dict = encoder_checkpoint['model_state_dict']
    encoder_state_dict.pop('policy_head.weight', None)
    encoder_state_dict.pop('policy_head.bias', None)
    encoder.load_state_dict(encoder_state_dict, strict=False)

    print(f"Loading temporal model for patching from: {args.checkpoint}")
    model = TemporalPolicyValueModel(
        encoder_model=encoder,
        policy_size=get_action_space_size(),
        d_model=config_params['EMBED_DIM']
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    print("Model loaded successfully.")

    # --- 2. Freeze Layers ---
    print("Freezing model layers for surgery...")
    for name, param in model.named_parameters():
        if 'policy_head' in name:
            print(f"  - Unfrozen: {name}")
            param.requires_grad = True
        else:
            param.requires_grad = False

    # --- 3. Load the Corrective Dataset ---
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[FATAL] Corrective dataset not found at: {dataset_path}")
        sys.exit(1)
    
    print(f"Loading corrective dataset from: {dataset_path}")
    corrective_data = torch.load(dataset_path)

    # --- 4. The "Surgical" Training Loop ---
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    model.train()
    print(f"\nStarting corrective training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        random.shuffle(corrective_data)
        total_loss = 0
        
        for data_point in tqdm(corrective_data, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False):
            # <<< FIXED: Calling our new, local batch preparation function.
            gnn_batch, cnn_batch, target_policies = prepare_corrective_batch([data_point], device)

            optimizer.zero_grad()
            
            policy_logits, _ = model(gnn_batch, cnn_batch)

            loss = -(torch.nn.functional.log_softmax(policy_logits, dim=1) * target_policies).sum(dim=1).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(corrective_data)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} complete. Average Loss: {avg_loss:.6f}")

    # --- 5. Save the Patched Model ---
    patched_model_filename = "temporal_checkpoint_patched_v1.pth.tar"
    save_path = paths.checkpoints_dir / patched_model_filename
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'patched_from': args.checkpoint,
        'patch_dataset': args.dataset
    }, save_path)
    
    print("\n✅ Surgical patch complete. New model saved to:")
    print(save_path)


if __name__ == "__main__":
    main()