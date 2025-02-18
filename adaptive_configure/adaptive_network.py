import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import json
import h5py
from tqdm.auto import tqdm
sys.path.append(os.path.dirname(os.getcwd())+"/scripts/rsl_rl")
from torch.utils.tensorboard import SummaryWriter

class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim=12*3 + 12, hidden_dim=512, output_dim=12*18 + 16):
        """
        Adaptive MLP model for estimating dynamic joint descriptions & general policy state.
        
        Args:
            input_dim: Flattened input dimension (12 joints * 3 states + 12 targets)
            hidden_dim: Hidden layer size
            output_dim: Flattened output dimension (12 joints * 18 descriptions + 16 policy state)
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, joint_states, targets):
        """
        Forward pass of the model.
        
        Args:
            joint_states: Tensor of shape [batch_size, 200, 12, 3]
            targets: Tensor of shape [batch_size, 200, 12]
        
        Returns:
            dynamic_joint_description: Tensor of shape [batch_size, 200, 12, 18]
            general_policy_state: Tensor of shape [batch_size, 200, 16]
        """
        batch_size = joint_states.shape[0]

        # Flatten inputs along the joint dimension
        x = torch.cat([joint_states.view(batch_size, 200, -1), targets.view(batch_size, 200, -1)], dim=-1)
        
        # Pass through MLP
        x = self.mlp(x)

        # Reshape outputs
        dynamic_joint_description = x[:, :, :12*18].view(batch_size, 200, 12, 18)
        general_policy_state = x[:, :, 12*18:].view(batch_size, 200, 16)

        return dynamic_joint_description, general_policy_state

def data_loader(dynamic_joint_state, targets, dynamic_joint_description, general_policy_state, batch_size=4, device='cuda'):
    """
    Prepare and batch the dataset.
    
    Args:
        dynamic_joint_state: Numpy array [200, 4096, 12, 3]
        targets: Numpy array [200, 4096, 12]
        dynamic_joint_description: Numpy array [200, 4096, 12, 18] (Ground Truth)
        general_policy_state: Numpy array [200, 4096, 16] (Ground Truth)
        device: Torch device ('cuda' or 'cpu')
    
    Returns:
        DataLoader for training
    """
    # Convert to PyTorch tensors
    dynamic_joint_state = torch.FloatTensor(dynamic_joint_state).permute(1, 0, 2, 3).to(device)  # [4096, 200, 12, 3]
    targets = torch.FloatTensor(targets).permute(1, 0, 2).to(device)  # [4096, 200, 12]
    dynamic_joint_description = torch.FloatTensor(dynamic_joint_description).permute(1, 0, 2, 3).to(device)  # [4096, 200, 12, 18]
    general_policy_state = torch.FloatTensor(general_policy_state).permute(1, 0, 2).to(device)  # [4096, 200, 16]

    # Create dataset and dataloader
    dataset = TensorDataset(dynamic_joint_state, targets, dynamic_joint_description, general_policy_state)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)  # Small batch size to avoid VRAM issues

    return dataloader

def train_model(model, train_loader, num_epochs=5, device='cuda', log_dir='logs'):
    """
    Train the AdaptiveMLP model.
    
    Args:
        model: Instance of AdaptiveMLP
        train_loader: DataLoader for training
        num_epochs: Number of epochs
        device: 'cuda' or 'cpu'
    
    Returns:
        Trained model
    """
    model.train()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Create directory for logging both training loss and model parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, log_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Tensorboard logger writer
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        total_loss = 0
        num_batches = 0

        for joint_states, targets, gt_dynamic_joint_description, gt_general_policy_state in train_loader:
            optimizer.zero_grad()

            # Forward pass
            dynamic_joint_description, general_policy_state = model(joint_states, targets)

            # Compute loss
            loss = criterion(dynamic_joint_description, gt_dynamic_joint_description) + \
                   criterion(general_policy_state, gt_general_policy_state)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # Log loss to TensorBoard
        writer.add_scalar("Training Loss", avg_loss, epoch)
    
    # Close TensorBoard writer
    writer.close()

    # Save trained model
    model_file = os.path.join(model_dir, "adaptive_configure_model.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

    return model

if __name__ == "__main__":
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example data generation (replace with actual data)
    dynamic_joint_state = torch.randn(200, 4096, 12, 3).numpy()
    targets = torch.randn(200, 4096, 12).numpy()
    dynamic_joint_description = torch.randn(200, 4096, 12, 18).numpy()
    general_policy_state = torch.randn(200, 4096, 16).numpy()

    # Prepare data
    train_loader = data_loader(dynamic_joint_state, targets, dynamic_joint_description, general_policy_state, device)

    # Initialize and train the model
    model = AdaptiveMLP().to(device)
    model = train_model(model, train_loader, num_epochs=5, device=device)
