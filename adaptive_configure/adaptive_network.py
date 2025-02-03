import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveConfigureNet(nn.Module):
    def __init__(self, obs_dim=268, action_dim=12, config_dim=18, hidden_dim=256):
        super().__init__()
        
        # Encoder for processing observation-action pairs
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Temporal attention to focus on important timesteps
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Configuration predictor for each joint
        self.config_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 12 * config_dim)  # 12 joints, each with config_dim parameters
        )

    def forward(self, obs, actions):
        batch_size, seq_len, _ = obs.shape
        
        # Concatenate observations and actions
        x = torch.cat([obs, actions], dim=-1)
        
        # Encode each timestep
        x = self.encoder(x)
        
        # Apply temporal attention
        attn_out, _ = self.temporal_attention(x, x, x)
        
        # Pool across temporal dimension using mean
        x = attn_out.mean(dim=1)
        
        # Predict configuration for all joints
        config = self.config_predictor(x)
        
        # Reshape to match target shape [batch_size, 12, config_dim]
        config = config.view(batch_size, 12, -1)
        
        return config

class ConfigureTrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_step(self, obs, actions, target_config):
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_config = self.model(obs, actions)
        
        # Compute MSE loss
        loss = F.mse_loss(pred_config, target_config)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def prepare_batch(obs, actions, config, device):
    """
    Prepare a batch of data for training
    
    Args:
        obs: numpy array of shape (num_episodes, seq_len, obs_dim)
        actions: numpy array of shape (num_episodes, seq_len, action_dim)
        config: numpy array of shape (config_joints, config_dim)
        device: torch device
    
    Returns:
        Tuple of tensors ready for training
    """
    # Convert to torch tensors
    obs = torch.FloatTensor(obs).to(device)
    actions = torch.FloatTensor(actions).to(device)
    
    # Convert config to tensor and ensure correct shape [batch_size, 12, 18]
    config = torch.FloatTensor(config).to(device)
    if len(config.shape) == 2:  # If config is [12, 18]
        config = config.unsqueeze(0).expand(obs.shape[0], -1, -1)
    
    return obs, actions, config

def train_model(model, train_obs, train_actions, train_configs, 
                num_epochs=100, batch_size=32, device='cuda'):
    """
    Train the adaptive configure network
    
    Args:
        model: AdaptiveConfigureNet instance
        train_obs: List of observation arrays for different robots
        train_actions: List of action arrays for different robots
        train_configs: List of configuration arrays for different robots
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
    """
    from tqdm.auto import tqdm
    trainer = ConfigureTrainer(model)
    model.train()
    
    # Create epoch progress bar without position specification
    epoch_iterator = tqdm(range(num_epochs), desc='Training Progress', leave=True)
    
    for epoch in epoch_iterator:
        total_loss = 0
        num_batches = 0
        
        # Train on each robot's data
        for obs, actions, config in zip(train_obs, train_actions, train_configs):
            # Prepare data
            obs_tensor, actions_tensor, config_tensor = prepare_batch(
                obs, actions, config, device
            )
            
            # Train in batches
            for i in range(0, len(obs), batch_size):
                batch_obs = obs_tensor[i:i+batch_size]
                batch_actions = actions_tensor[i:i+batch_size]
                batch_config = config_tensor[i:i+batch_size]
                
                loss = trainer.train_step(batch_obs, batch_actions, batch_config)
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        # Print epoch results on a new line
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
    
    return model

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = AdaptiveConfigureNet(
        obs_dim=268,  # From the data shape
        action_dim=12,  # From the data shape
        config_dim=18,  # From the data shape
    ).to(device)
    print("AdaptiveConfigureNet Structure:")
    print(model)
    
    # Training would be done by:
    # train_model(model, [robot1_obs, robot2_obs], 
    #                   [robot1_actions, robot2_actions],
    #                   [robot1_config, robot2_config])
