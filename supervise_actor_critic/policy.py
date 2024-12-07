from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal


class SuperviseActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        nr_dynamic_joint_observations,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ModifiedActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        # Precompute the input dimension for the actor after preprocessing
        # After slicing and reshaping. 
        # 3 * nr_dynamic_joint_observations means joint_pos rel, joint_vel_rel and action(previous)
        # 12 means base_lin_vel, base_ang_vel, projected_gravity and target_x_y_yaw_rel
        self.num_processed_actor_obs = 3 * nr_dynamic_joint_observations + 12 
        # nr_dynamic_joint_observations means the number of joints, which is also the number of actions
        self.num_actions = nr_dynamic_joint_observations

        # Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_processed_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], self.num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print(f"Actor MLP: {self.actor}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def preprocess_actor_input(self, dynamic_joint_state, general_policy_state):
        """
        Preprocess the dataset for the actor model:
        1. Slice the dataset to take only the first three elements along the last axis.
        2. Split along the last axis into three parts.
        3. Concatenate these parts along the second axis.

        Args:
            observations (torch.Tensor): Input tensor of shape (batch_num, 15, 6).

        Returns:
            torch.Tensor: Preprocessed tensor of shape (batch_num, 15 * 3).
        """
        sliced_1 = general_policy_state[:, :12] # base_lin_vel, base_ang_vel, projected_gravity and target_x_y_yaw_rel from general_policy_state
        swapped = torch.cat([sliced_1[:, :6], sliced_1[:, 9:12], sliced_1[:, 6:9]], dim=-1) # base_lin_vel, base_ang_vel, arget_x_y_yaw_rel and projected_gravity
        
        sliced_2 = dynamic_joint_state  # dynamic_joint_state (batch_num, 15, 3)
        split = torch.split(sliced_2, 1, dim=-1)  # Split into 3 tensors along last axis
        concatenated = torch.cat(split, dim=1).squeeze(-1)  # Concatenate along the second axis. This is joint_pos_rel, joint_vel_rel and action(previous)
        
        # The order of a actor_critic model obs input is base_lin_vel, base_ang_vel, projected_gravity, target_x_y_yaw_rel, joint_pos_rel, joint_vel_rel and action(previous)
        return torch.cat([swapped, concatenated], dim=1) 

    def forward(self, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, general_policy_state):
        """
        Forward pass for the actor.

        Args:
            observations (torch.Tensor): Input tensor of shape (batch_num, 15, 6).

        Returns:
            torch.Tensor: Output tensor of shape (batch_num, num_actions).
        """
        preprocessed_input = self.preprocess_actor_input(dynamic_joint_state, general_policy_state)  # Preprocess input
        return self.actor(preprocessed_input)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("Invalid activation function!")
        return None
    
def get_policy(nr_dynamic_joint_observations: int, model_device: str):
    policy = SuperviseActorCritic(nr_dynamic_joint_observations)
    policy = torch.jit.script(policy)
    policy.to(model_device)

    return policy