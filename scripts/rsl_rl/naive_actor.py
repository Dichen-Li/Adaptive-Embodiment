from torch import nn
class ActorMLP(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=128, output_dim=15, activation=nn.ELU()):
        """
        ActorMLP is a sequential feedforward neural network.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of units in each hidden layer.
            output_dim (int): Number of output features.
            activation (nn.Module): Activation function to use between layers.
        """
        super(ActorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Forward pass of the ActorMLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.network(x)