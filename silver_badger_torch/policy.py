import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# import ipdb

class Policy(nn.Module):
    def __init__(self, initial_softmax_temperature,
                 softmax_temperature_min, stability_epsilon, policy_mean_abs_clip, policy_std_min_clip,
                 policy_std_max_clip):
        super(Policy, self).__init__()
        self.softmax_temperature_min = softmax_temperature_min
        self.stability_epsilon = stability_epsilon
        self.policy_mean_abs_clip = policy_mean_abs_clip
        self.policy_std_min_clip = policy_std_min_clip
        self.policy_std_max_clip = policy_std_max_clip

        # Constants
        dynamic_joint_des_dim = 18
        general_state_dim = 16
        dynamic_joint_state_dim = 3

        # hyper param
        scale_factor = 2

        dynamic_joint_state_mask_dim = 64
        dynamic_joint_state_feat = 4 * scale_factor
        self.dynamic_joint_state_mask1 = nn.Linear(dynamic_joint_des_dim, dynamic_joint_state_mask_dim)
        self.dynamic_joint_layer_norm = nn.LayerNorm(dynamic_joint_state_mask_dim, eps=1e-6)
        self.dynamic_joint_state_mask2 = nn.Linear(dynamic_joint_state_mask_dim, dynamic_joint_state_mask_dim)
        self.joint_log_softmax_temperature = nn.Parameter(torch.tensor([initial_softmax_temperature - self.softmax_temperature_min]).log())
        self.latent_dynamic_joint_state = nn.Linear(dynamic_joint_state_dim, dynamic_joint_state_feat)

        combined_action_feat_dim = dynamic_joint_state_mask_dim * dynamic_joint_state_feat + general_state_dim
        action_latent_dims = [512, 256, 128 * scale_factor]
        self.action_latent1 = nn.Linear(combined_action_feat_dim, action_latent_dims[0])
        self.action_layer_norm = nn.LayerNorm(action_latent_dims[0], eps=1e-6)
        self.action_latent2 = nn.Linear(action_latent_dims[0], action_latent_dims[1])
        self.action_latent3 = nn.Linear(action_latent_dims[1], action_latent_dims[2])

        action_des_latent_dim = 128 * scale_factor
        self.action_description_latent1 = nn.Linear(dynamic_joint_des_dim, action_des_latent_dim)
        self.action_description_layer_norm = nn.LayerNorm(action_des_latent_dim, eps=1e-6)
        self.action_description_latent2 = nn.Linear(action_des_latent_dim, action_des_latent_dim)

        policy_in_dim = dynamic_joint_state_feat + action_latent_dims[-1] + action_des_latent_dim
        policy_hidden_dim = 128 * scale_factor
        self.policy_mean_layer1 = nn.Linear(policy_in_dim, policy_hidden_dim)
        self.policy_mean_layer_norm = nn.LayerNorm(policy_hidden_dim, eps=1e-6)
        self.policy_mean_layer2 = nn.Linear(policy_hidden_dim, 1)
        self.policy_logstd_layer = nn.Linear(policy_hidden_dim, 1)

    def forward(self, dynamic_joint_description, dynamic_joint_state, general_state):
        dynamic_joint_state_mask = self.dynamic_joint_state_mask1(dynamic_joint_description)
        dynamic_joint_state_mask = F.elu(self.dynamic_joint_layer_norm(dynamic_joint_state_mask))
        dynamic_joint_state_mask = torch.tanh(self.dynamic_joint_state_mask2(dynamic_joint_state_mask))
        dynamic_joint_state_mask = torch.clamp(dynamic_joint_state_mask,
                                               -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)

        latent_dynamic_joint_state = F.elu(self.latent_dynamic_joint_state(dynamic_joint_state))

        joint_e_x = torch.exp(dynamic_joint_state_mask / (torch.exp(self.joint_log_softmax_temperature) + self.softmax_temperature_min))
        dynamic_joint_state_mask = joint_e_x / (joint_e_x.sum(dim=-1, keepdim=True) + self.stability_epsilon)
        dynamic_joint_state_mask = dynamic_joint_state_mask.unsqueeze(-1).repeat(1, 1, 1, latent_dynamic_joint_state.size(-1))
        masked_dynamic_joint_state = dynamic_joint_state_mask * latent_dynamic_joint_state.unsqueeze(-2)
        masked_dynamic_joint_state = masked_dynamic_joint_state.view(masked_dynamic_joint_state.shape[:-2] + (masked_dynamic_joint_state.shape[-2] * masked_dynamic_joint_state.shape[-1],))
        dynamic_joint_latent = masked_dynamic_joint_state.sum(dim=-2)

        combined_input = torch.cat([dynamic_joint_latent, general_state], dim=-1)

        action_latent = self.action_latent1(combined_input)
        action_latent = F.elu(self.action_layer_norm(action_latent))
        action_latent = F.elu(self.action_latent2(action_latent))
        action_latent = self.action_latent3(action_latent)

        action_description_latent = self.action_description_latent1(dynamic_joint_description)
        action_description_latent = F.elu(self.action_description_layer_norm(action_description_latent))
        action_description_latent = self.action_description_latent2(action_description_latent)

        action_latent = action_latent.unsqueeze(-2).repeat(1, action_description_latent.size(-2), 1)
        combined_action_latent = torch.cat([action_latent, latent_dynamic_joint_state.detach(), action_description_latent], dim=-1)

        policy_mean = self.policy_mean_layer1(combined_action_latent)
        policy_mean = F.elu(self.policy_mean_layer_norm(policy_mean))
        policy_mean = self.policy_mean_layer2(policy_mean)
        policy_mean = torch.clamp(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)

        return policy_mean.squeeze(-1)


def get_policy(model_device: str):
    initial_softmax_temperature = 1.0
    softmax_temperature_min = 0.015
    stability_epsilon = 0.00000001
    policy_mean_abs_clip = 10.0  # 10.0. This value should be adjusted based on data? Or the data should be normalized.
    policy_std_min_clip = 0.00000001
    policy_std_max_clip = 2.0

    policy = Policy(initial_softmax_temperature, softmax_temperature_min, stability_epsilon, policy_mean_abs_clip, policy_std_min_clip, policy_std_max_clip)
    # policy = torch.jit.script(policy)
    policy.to(model_device)

    # # Load weights
    # policy.dynamic_joint_state_mask1.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_0_kernel.npy"))).T
    # policy.dynamic_joint_state_mask1.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_0_bias.npy")))
    # policy.dynamic_joint_layer_norm.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_0_scale.npy")))
    # policy.dynamic_joint_layer_norm.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_0_bias.npy")))
    # policy.dynamic_joint_state_mask2.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_1_kernel.npy"))).T
    # policy.dynamic_joint_state_mask2.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_1_bias.npy")))
    # policy.latent_dynamic_joint_state.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_2_kernel.npy"))).T
    # policy.latent_dynamic_joint_state.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_2_bias.npy")))
    # policy.dynamic_foot_state_mask1.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_3_kernel.npy"))).T
    # policy.dynamic_foot_state_mask1.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_3_bias.npy")))
    # policy.dynamic_foot_layer_norm.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_1_scale.npy")))
    # policy.dynamic_foot_layer_norm.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_1_bias.npy")))
    # policy.dynamic_foot_state_mask2.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_4_kernel.npy"))).T
    # policy.dynamic_foot_state_mask2.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_4_bias.npy")))
    # policy.latent_dynamic_foot_state.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_5_kernel.npy"))).T
    # policy.latent_dynamic_foot_state.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_5_bias.npy")))
    # policy.action_latent1.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_6_kernel.npy"))).T
    # policy.action_latent1.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_6_bias.npy")))
    # policy.action_layer_norm.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_2_scale.npy")))
    # policy.action_layer_norm.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_2_bias.npy")))
    # policy.action_latent2.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_7_kernel.npy"))).T
    # policy.action_latent2.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_7_bias.npy")))
    # policy.action_latent3.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_8_kernel.npy"))).T
    # policy.action_latent3.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_8_bias.npy")))
    # policy.action_description_latent1.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_9_kernel.npy"))).T
    # policy.action_description_latent1.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_9_bias.npy")))
    # policy.action_description_layer_norm.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_3_scale.npy")))
    # policy.action_description_layer_norm.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_3_bias.npy")))
    # policy.action_description_latent2.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_10_kernel.npy"))).T
    # policy.action_description_latent2.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_10_bias.npy")))
    # policy.policy_mean_layer1.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_11_kernel.npy"))).T
    # policy.policy_mean_layer1.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_11_bias.npy")))
    # policy.policy_mean_layer_norm.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_4_scale.npy")))
    # policy.policy_mean_layer_norm.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/LayerNorm_4_bias.npy")))
    # policy.policy_mean_layer2.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_12_kernel.npy"))).T
    # policy.policy_mean_layer2.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_12_bias.npy")))
    # policy.policy_logstd_layer.weight.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_13_kernel.npy"))).T
    # policy.policy_logstd_layer.bias.data = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "jax_nn_weights/Dense_13_bias.npy")))

    return policy


if __name__ == "__main__":
    # define the device = 'cuda:0'
    model_device = 'cuda:0'

    policy = get_policy(model_device)

    dummy_dynamic_joint_description = torch.zeros((1, 13, 18), device=model_device, dtype=torch.float32)
    dummy_dynamic_joint_state = torch.zeros((1, 13, 3), device=model_device, dtype=torch.float32)
    dummy_dynamic_foot_description = torch.zeros((1, 4, 10), device=model_device, dtype=torch.float32)
    dummy_dynamic_foot_state = torch.zeros((1, 4, 2), device=model_device, dtype=torch.float32)
    dummy_general_policy_state = torch.zeros((1, 16), device=model_device, dtype=torch.float32)

    import time

    nr_evals = 1_000
    start = time.time()
    for i in range(nr_evals):
        with torch.no_grad():
            action = policy(dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_general_policy_state)
    end = time.time()
    print("Average time per evaluation: ", (end - start) / nr_evals)
