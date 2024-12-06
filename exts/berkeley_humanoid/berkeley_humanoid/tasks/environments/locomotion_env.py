from __future__ import annotations
import torch
import math

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.core.utils.torch.rotations import compute_rot
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

        self.nominal_trunk_z = self.robot.data.default_root_state[0, 2]
        self.joint_nominal_positions = self.robot.data.default_joint_pos
        self.action_scaling_factor = self.cfg.action_scaling_factor
        self.p_gains = self.robot.actuators["base_legs"].stiffness
        self.d_gains = self.robot.actuators["base_legs"].damping

        self.step_sampling_probability = self.cfg.step_sampling_probability

        self.goal_velocities = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.sim.device)
        self.previous_actions = torch.zeros((self.num_envs, self.nr_joints), dtype=torch.float32, device=self.sim.device)

        self.reward_curriculum_steps = self.cfg.reward_curriculum_steps
        self.tracking_xy_velocity_command_coeff = self.cfg.tracking_xy_velocity_command_coeff
        self.tracking_yaw_velocity_command_coeff = self.cfg.tracking_yaw_velocity_command_coeff
        self.z_velocity_coeff = self.cfg.z_velocity_coeff
        self.pitch_roll_vel_coeff = self.cfg.pitch_roll_vel_coeff
        self.pitch_roll_pos_coeff = self.cfg.pitch_roll_pos_coeff
        self.actuator_joint_nominal_diff_coeff = self.cfg.actuator_joint_nominal_diff_coeff
        self.actuator_joint_nominal_diff_joints = self.cfg.actuator_joint_nominal_diff_joints
        self.joint_position_limit_coeff = self.cfg.joint_position_limit_coeff
        self.joint_acceleration_coeff = self.cfg.joint_acceleration_coeff
        self.joint_torque_coeff = self.cfg.joint_torque_coeff
        self.action_rate_coeff = self.cfg.action_rate_coeff
        self.base_height_coeff = self.cfg.base_height_coeff
        self.air_time_coeff = self.cfg.air_time_coeff
        self.symmetry_air_coeff = self.cfg.symmetry_air_coeff

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


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        trunk_contact_sensor = self.scene.sensors[self.trunk_contact_cfg.name]
        trunk_contact_forces = trunk_contact_sensor.data.net_forces_w_history
        trunk_contact = torch.any(torch.max(torch.norm(trunk_contact_forces[:, :, self.trunk_contact_cfg.body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        terminated = trunk_contact

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated


    def _get_rewards(self) -> torch.Tensor:
        global_step = self.common_step_counter * self.num_envs
        curriculum_coeff = min(global_step / self.reward_curriculum_steps, 1.0)

        total_reward = compute_rewards(
            curriculum_coeff,
            self.tracking_xy_velocity_command_coeff,
            self.tracking_yaw_velocity_command_coeff,
            self.z_velocity_coeff,
            self.pitch_roll_vel_coeff,
            self.pitch_roll_pos_coeff,
            self.actuator_joint_nominal_diff_coeff,
            self.actuator_joint_nominal_diff_joints,
            self.joint_position_limit_coeff,
            self.joint_acceleration_coeff,
            self.joint_torque_coeff,
            self.action_rate_coeff,
            self.base_height_coeff,
            self.air_time_coeff,
            self.symmetry_air_coeff,
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.root_state_w[:, :3],
            self.robot.data.root_state_w[:, 3:7],
            self.robot.data.joint_pos,
            self.joint_nominal_positions,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.actions,
            self.previous_actions,
            self.robot.data.joint_acc,
            self.robot.data.applied_torque,
            self.goal_velocities,
            self.nominal_trunk_z
        )

        self.previous_actions = self.actions.clone()

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
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    # From: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L530

    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )

    return quaternions[..., 1:] / sin_half_angles_over_angles


@torch.jit.script
def compute_rewards(
    curriculum_coeff: float,
    tracking_xy_velocity_command_coeff: float,
    tracking_yaw_velocity_command_coeff: float,
    z_velocity_coeff: float,
    pitch_roll_vel_coeff: float,
    pitch_roll_pos_coeff: float,
    actuator_joint_nominal_diff_coeff: float,
    actuator_joint_nominal_diff_joints: list[int],
    joint_position_limit_coeff: float,
    joint_acceleration_coeff: float,
    joint_torque_coeff: float,
    action_rate_coeff: float,
    base_height_coeff: float,
    air_time_coeff: float,
    symmetry_air_coeff: float,
    current_local_linear_velocity: torch.Tensor,
    current_local_angular_velocity: torch.Tensor,
    linear_position: torch.Tensor,
    orientation_quat: torch.Tensor,
    joint_positions: torch.Tensor,
    joint_nominal_positions: torch.Tensor,
    joint_position_soft_lower_limits: torch.Tensor,
    joint_position_soft_upper_limits: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    joint_accelerations: torch.Tensor,
    joint_torques: torch.Tensor,
    goal_velocities: torch.Tensor,
    nominal_trunk_z: float
) -> torch.Tensor:
    
    # Tracking xy velocity command reward
    current_local_linear_velocity_xy = current_local_linear_velocity[:, :2]
    desired_local_linear_velocity_xy = goal_velocities[:, :2]
    xy_velocity_difference_norm = torch.sum(torch.square(current_local_linear_velocity_xy - desired_local_linear_velocity_xy), dim=1)
    tracking_xy_velocity_command_reward = tracking_xy_velocity_command_coeff * torch.exp(-xy_velocity_difference_norm / 0.25)

    # Tracking angular velocity command reward
    current_local_yaw_velocity = current_local_angular_velocity[:, 2]
    desired_local_yaw_velocity = goal_velocities[:, 2]
    yaw_velocity_difference_norm = torch.square(current_local_yaw_velocity - desired_local_yaw_velocity)
    tracking_yaw_velocity_command_reward = tracking_yaw_velocity_command_coeff * torch.exp(-yaw_velocity_difference_norm / 0.25)

    # Linear velocity reward
    z_velocity_squared = torch.square(current_local_linear_velocity[:, 2])
    linear_velocity_reward = curriculum_coeff * z_velocity_coeff * z_velocity_squared

    # Angular velocity reward
    angular_velocity_norm = torch.sum(torch.square(current_local_angular_velocity[:, :2]), dim=1)
    angular_velocity_reward = curriculum_coeff * pitch_roll_vel_coeff * angular_velocity_norm

    # Angular position reward
    orientation_euler = quaternion_to_axis_angle(orientation_quat)
    pitch_roll_position_norm = torch.sum(torch.square(orientation_euler[:, :2]), dim=1)
    angular_position_reward = curriculum_coeff * pitch_roll_pos_coeff * pitch_roll_position_norm

    # Joint nominal position difference reward
    actuator_joint_nominal_diff_norm = torch.sum(torch.square(joint_positions[:, actuator_joint_nominal_diff_joints] - joint_nominal_positions[:, actuator_joint_nominal_diff_joints]), dim=1)
    actuator_joint_nominal_diff_reward = curriculum_coeff * actuator_joint_nominal_diff_coeff * actuator_joint_nominal_diff_norm

    # Joint position limit reward
    lower_limit_penalty = -torch.minimum(joint_positions - joint_position_soft_lower_limits, torch.tensor(0.0, device=joint_positions.device)).sum(dim=1)
    upper_limit_penalty = torch.maximum(joint_positions - joint_position_soft_upper_limits, torch.tensor(0.0, device=joint_positions.device)).sum(dim=1)
    joint_position_limit_reward = curriculum_coeff * joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)

    # Joint acceleration reward
    acceleration_norm = torch.sum(torch.square(joint_accelerations), dim=1)
    acceleration_reward = curriculum_coeff * joint_acceleration_coeff * acceleration_norm

    # Joint torque reward
    torque_norm = torch.sum(torch.square(joint_torques), dim=1)
    torque_reward = curriculum_coeff * joint_torque_coeff * torque_norm

    # Action rate reward
    action_rate_norm = torch.sum(torch.square(actions - previous_actions), dim=1)
    action_rate_reward = curriculum_coeff * action_rate_coeff * action_rate_norm

    # Walking height reward
    trunk_z = linear_position[:, 2]
    height_difference_squared = (trunk_z - nominal_trunk_z) ** 2
    base_height_reward = curriculum_coeff * base_height_coeff * height_difference_squared

    # Air time reward
    # TODO:

    # Symmetry air reward
    # TODO:

    # Total reward
    tracking_reward = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward
    reward_penalty = linear_velocity_reward + angular_velocity_reward + angular_position_reward + actuator_joint_nominal_diff_reward + \
                     joint_position_limit_reward + acceleration_reward + torque_reward + action_rate_reward + \
                     base_height_reward + air_time_reward + symmetry_air_reward
    reward = tracking_reward + reward_penalty
    reward = torch.maximum(reward, torch.tensor(0.0, device=reward.device))

    return reward
