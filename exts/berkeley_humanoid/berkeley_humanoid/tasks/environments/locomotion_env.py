from __future__ import annotations
import torch
import math

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sensors import ContactSensor
import omni.isaac.lab.sim as sim_utils


class LocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg


    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.trunk_contact_cfg = self.cfg.trunk_contact_cfg
        self.trunk_contact_cfg.resolve(self.scene)
        self.feet_contact_cfg = self.cfg.feet_contact_cfg
        self.feet_contact_cfg.resolve(self.scene)

        self.nr_joints = self.robot.data.default_joint_pos.shape[1]
        self.nr_feet = self.cfg.nr_feet

        self.joint_nominal_positions = self.robot.data.default_joint_pos
        self.action_scaling_factor = self.cfg.action_scaling_factor
        self.p_gains = self.robot.actuators["base_legs"].stiffness
        self.d_gains = self.robot.actuators["base_legs"].damping

        self.step_sampling_probability = self.cfg.step_sampling_probability
        self.goal_velocities = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.sim.device)

        self.set_observation_indices()


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions


    def _apply_action(self):
        scaled_actions = self.actions * self.action_scaling_factor
        target_joint_positions = self.joint_nominal_positions + scaled_actions
        torques = self.p_gains * (target_joint_positions - self.robot.data.joint_pos) - self.d_gains * self.robot.data.joint_vel

        self.robot.set_joint_effort_target(torques)

        self.prev_actions = self.actions.clone()


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        trunk_contact_sensor = self.scene.sensors[self.trunk_contact_cfg.name]
        trunk_contact_forces = trunk_contact_sensor.data.net_forces_w_history
        trunk_contact = torch.any(torch.max(torch.norm(trunk_contact_forces[:, :, self.trunk_contact_cfg.body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        terminated = trunk_contact

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated


    def _get_rewards(self) -> torch.Tensor:
        lin_vel = self.robot.data.root_lin_vel_b

        total_reward = compute_rewards(lin_vel, self.goal_velocities[:, :2])

        return total_reward
    

    def set_observation_indices(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = [current_observation_idx + i for i in range(self.nr_joints)]
        current_observation_idx += self.nr_joints

        self.joint_velocities_obs_idx = [current_observation_idx + i for i in range(self.nr_joints)]
        current_observation_idx += self.nr_joints

        self.joint_previous_actions_obs_idx = [current_observation_idx + i for i in range(self.nr_joints)]
        current_observation_idx += self.nr_joints

        self.feet_contact_obs_idx = [current_observation_idx + i for i in range(self.nr_feet)]
        current_observation_idx += self.nr_feet

        self.feet_air_time_obs_idx = [current_observation_idx + i for i in range(self.nr_feet)]
        current_observation_idx += self.nr_feet

        self.trunk_linear_velocity_obs_idx = [current_observation_idx + i for i in range(3)]
        current_observation_idx += 3

        self.trunk_angular_velocity_obs_idx = [current_observation_idx + i for i in range(3)]
        current_observation_idx += 3

        self.goal_velocities_obs_idx = [current_observation_idx + i for i in range(3)]
        current_observation_idx += 3

        self.projected_gravity_vector_obs_idx = [current_observation_idx + i for i in range(3)]
        current_observation_idx += 3

        self.height_obs_idx = [current_observation_idx]
        current_observation_idx += 1


    def _get_observations(self) -> dict:
        # Joint-specific observations
        joint_positions = self.robot.data.joint_pos - self.joint_nominal_positions
        joint_velocities = self.robot.data.joint_vel
        joint_previous_actions = self.actions

        # Feet-specific observations
        feet_contact_sensors = self.scene.sensors[self.feet_contact_cfg.name]
        feet_contact_forces = feet_contact_sensors.data.net_forces_w_history
        feet_contacts = (torch.max(torch.norm(feet_contact_forces[:, :, self.feet_contact_cfg.body_ids], dim=-1), dim=1)[0] > 1.0).float()
        feet_air_times = feet_contact_sensors.data.last_air_time[:, self.feet_contact_cfg.body_ids]

        # General observations
        trunk_linear_velocity = self.robot.data.root_lin_vel_b
        trunk_angular_velocity = self.robot.data.root_ang_vel_b

        should_sample_new_goal_velocities = torch.rand((self.num_envs,), device=self.sim.device) < self.step_sampling_probability
        self.goal_velocities[should_sample_new_goal_velocities] = torch.rand((should_sample_new_goal_velocities.sum(), 3), device=self.sim.device).uniform_(-1.0, 1.0)
        goal_velocities = self.goal_velocities

        projected_gravity_vector = self.robot.data.projected_gravity_b
        height = self.robot.data.root_state_w[:, [2]]

        observation = torch.cat(
            [
                joint_positions,
                joint_velocities,
                joint_previous_actions,
                feet_contacts,
                feet_air_times,
                trunk_linear_velocity,
                trunk_angular_velocity,
                goal_velocities,
                projected_gravity_vector,
                height,
            ],
            dim=1,
        )
        
        # TODO: Add noise

        # TODO: Dropout

        # Normalize & Clip
        observation[:, self.joint_positions_obs_idx] /= 4.6
        observation[:, self.joint_velocities_obs_idx] /= 35.0
        observation[:, self.joint_previous_actions_obs_idx] /= 10.0
        observation[:, self.feet_contact_obs_idx] = (observation[:, self.feet_contact_obs_idx] / 0.5) - 1.0
        observation[:, self.feet_air_time_obs_idx] = torch.clamp((observation[:, self.feet_air_time_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[:, self.trunk_linear_velocity_obs_idx] = torch.clamp(observation[:, self.trunk_linear_velocity_obs_idx] / 10.0, -1.0, 1.0)
        observation[:, self.trunk_angular_velocity_obs_idx] = torch.clamp(observation[:, self.trunk_angular_velocity_obs_idx] / 50.0, -1.0, 1.0)
        observation[:, self.height_obs_idx] = torch.clamp((observation[:, self.height_obs_idx] / 1.0) - 1.0, -1.0, 1.0)

        policy_observation = observation[:,
            self.joint_positions_obs_idx + \
            self.joint_velocities_obs_idx + \
            self.joint_previous_actions_obs_idx + \
            self.trunk_angular_velocity_obs_idx + \
            self.goal_velocities_obs_idx + \
            self.projected_gravity_vector_obs_idx
        ]

        return {"policy": policy_observation, "critic": observation}



### JIT compiled helper functions ###
@torch.jit.script
def compute_rewards(lin_vel: torch.Tensor, target_x_vel: torch.Tensor) -> torch.Tensor:
    lin_vel_error = torch.sum(torch.square(lin_vel[:, :2] - target_x_vel), dim=1)
    track_lin_vel_xy_exp = torch.exp(-lin_vel_error / math.sqrt(0.25) ** 2)

    total_reward = track_lin_vel_xy_exp

    return total_reward
