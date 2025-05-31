import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import chess

from gnn_sq_model import GNN_sq_BaseModel
from gnn_sq_input import get_gnn_sq_input, NUM_SQUARES, create_adjacency_matrix
from square_features import SquareFeatures # To get input_feature_dim

def create_synthetic_data(num_samples: int, input_feature_dim: int, gnn_embedding_dim: int):
    """
    Creates synthetic input data (node features, adj_matrix) and random target embeddings.
    """
    all_node_features = []
    all_adj_matrices = []
    all_target_embeddings = []

    # Use a fixed adjacency matrix for all samples for simplicity in this synthetic test
    # (though in reality, it's fixed anyway based on our current design)
    adj_matrix_np = create_adjacency_matrix(adjacency_type="8_way").astype(np.float32)
    adj_matrix_tensor = torch.from_numpy(adj_matrix_np)

    fen_strings = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Initial position
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", # After 1. e4 c5
        "rnbqkbnr/ppp2ppp/4p3/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq d6 0 3", # After 1. e4 e6 2. Nf3 d5
        "8/k7/8/8/8/8/K7/8 w - - 0 1", # King vs King
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" # Complex mid-game
    ]
    
    if num_samples > len(fen_strings):
        # If more samples requested, reuse FENs or add more unique ones
        fen_strings_to_use = (fen_strings * (num_samples // len(fen_strings) + 1))[:num_samples]
    else:
        fen_strings_to_use = fen_strings[:num_samples]

    for i in range(num_samples):
        board = chess.Board(fen_strings_to_use[i])
        node_features_np, _ = get_gnn_sq_input(board, adjacency_type="8_way") # Adj matrix already created

        all_node_features.append(torch.from_numpy(node_features_np))
        all_adj_matrices.append(adj_matrix_tensor) # Using the same adj matrix for all

        # Create random target embeddings for each square for this sample
        target = torch.randn(NUM_SQUARES, gnn_embedding_dim)
        all_target_embeddings.append(target)

    return all_node_features, all_adj_matrices, all_target_embeddings


def main():
    # Hyperparameters for the synthetic test
    num_synthetic_samples = 5
    input_feature_dim = SquareFeatures.get_feature_dimension() # Should be 17
    gnn_embedding_dim = 32  # Must match the GNN_sq_BaseModel's output
    learning_rate = 0.0002
    num_epochs = 1500 # Increase if loss doesn't converge enough

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create synthetic data
    node_features_list, adj_matrices_list, target_embeddings_list = \
        create_synthetic_data(num_synthetic_samples, input_feature_dim, gnn_embedding_dim)

    # Instantiate the model
    model = GNN_sq_BaseModel(input_feature_dim=input_feature_dim, 
                             gnn_embedding_dim=gnn_embedding_dim).to(device)
    model.train() # Set the model to training mode

    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print(f"Starting training on {num_synthetic_samples} synthetic samples for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(num_synthetic_samples):
            node_features = node_features_list[i].to(device)
            adj_matrix = adj_matrices_list[i].to(device) # adj_matrix is the same for all in this setup
            targets = target_embeddings_list[i].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            output_embeddings = model(node_features, adj_matrix)
            
            # Calculate loss
            loss = criterion(output_embeddings, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_synthetic_samples
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.8f}")

    print("Training finished.")
    print(f"Final average loss: {avg_loss:.8f}")

    # You can add a final check here:
    # After training, predict again and see if the output is very close to the target for one sample.
    with torch.no_grad():
        model.eval()
        sample_idx = 0
        node_features = node_features_list[sample_idx].to(device)
        adj_matrix = adj_matrices_list[sample_idx].to(device)
        expected_target = target_embeddings_list[sample_idx].to(device)
        
        predicted_embeddings = model(node_features, adj_matrix)
        final_sample_loss = criterion(predicted_embeddings, expected_target).item()
        print(f"Loss on one sample ({sample_idx}) after training: {final_sample_loss:.8f}")
        
        # Check how close one of the predicted embeddings is to its target
        # print("Target for first node of first sample:", expected_target[0, :5].cpu().numpy())
        # print("Predicted for first node of first sample:", predicted_embeddings[0, :5].cpu().numpy())

    if avg_loss < 0.01 : # Threshold for considering it "overfit" for this synthetic task
        print("Successfully overfit the small synthetic dataset!")
    else:
        print("Could not sufficiently overfit the synthetic dataset. Further checks might be needed.")

if __name__ == '__main__':
    main()