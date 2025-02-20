# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math

import omni.isaac.core.utils.torch as torch_utils
from berkeley_humanoid.tasks.direct.locomotion.command_function import RandomCommands
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

from .joint_position_controller import JointPositionAction

from typing import Optional

import os
import json

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def track_vel_exp(curr, target, std=math.sqrt(0.25)):
    """
    Adapted from
    https://isaac-sim.github.io/IsaacLab/main/_modules/omni/isaac/lab/envs/mdp/rewards.html#track_lin_vel_xy_exp
    Reward tracking of velocity commands using exponential kernel.
    """
    lin_vel_error = torch.sum(torch.square(curr - target), dim=1)
    lin_vel_error = torch.exp(-lin_vel_error / std ** 2)
    return lin_vel_error


class LocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        # self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        # self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(self.cfg.controlled_joints)

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # some transfered stuff
        self.command_generator = RandomCommands(self, self.cfg.x_vel_range, self.cfg.y_vel_range,
                                                self.cfg.yaw_vel_range, self.cfg.resampling_interval)
        self.prev_actions = None

        # configs needed for
        self.reward_cfgs = self.cfg.reward_cfgs
        for cfg in self.reward_cfgs.values():
            cfg.resolve(self.scene)

        self.controller = JointPositionAction(self, self.cfg.action_scale,
                                              use_default_offset=self.cfg.controller_use_offset)
        
        ## Define the one policy initialized parameters below
        # Get the position for self.relative_joint_position_normalized, axis for self.relative_joint_axis_local and bbox for self.robot_dimensions
        self.get_robot_description_vec_json()

        self.joint_names = self.robot.data.joint_names
        self.foot_names = self.robot.data.body_names
        
        multi_robot_max_observation_size = -1
        self.joint_nr_direct_child_joints = [int("foot" in name) for name in self.foot_names]
        self.name_to_description_vector = self.get_name_to_description_vector()

        self.initial_observation = self.get_initial_observation(multi_robot_max_observation_size)

        # Get the dictionary of name to id
        self.observation_name_to_id = self.get_observation_space(multi_robot_max_observation_size)
        # Idx that need to be updated every step
        self.joint_positions_update_obs_idx = [self.observation_name_to_id[joint_name + "_position"] for joint_name in self.joint_names]
        self.joint_velocities_update_obs_idx = [self.observation_name_to_id[joint_name + "_velocity"] for joint_name in self.joint_names]
        self.joint_previous_actions_update_obs_idx = [self.observation_name_to_id[joint_name + "_previous_action"] for joint_name in self.joint_names]
        self.foot_ground_contact_update_obs_idx = [self.observation_name_to_id[foot_name + "_ground_contact"] for foot_name in self.foot_names if "foot" in foot_name]
        self.foot_time_since_last_ground_contact_update_obs_idx = [self.observation_name_to_id[foot_name + "_cycles_since_last_ground_contact"] for foot_name in self.foot_names if "foot" in foot_name]
        self.trunk_linear_vel_update_obs_idx = [self.observation_name_to_id["trunk_" + observation_name] for observation_name in ["x_velocity", "y_velocity", "z_velocity"]]
        self.trunk_angular_vel_update_obs_idx = [self.observation_name_to_id["trunk_" + observation_name] for observation_name in ["roll_velocity", "pitch_velocity", "yaw_velocity"]]
        self.goal_velocity_update_obs_idx = [self.observation_name_to_id["goal_" + observation_name] for observation_name in ["x_velocity", "y_velocity", "yaw_velocity"]]
        self.projected_gravity_update_obs_idx = [self.observation_name_to_id["projected_gravity_" + observation_name] for observation_name in ["x", "y", "z"]]
        self.height_update_obs_idx = [self.observation_name_to_id["height_0"]]
        ## Define the one policy initialized parameters above
        print("LocomotionEnv init done")

    def _setup_scene(self):
        """
        Please note that assets need to be manually instantiated here.
        Please add a few lines every time you modify cfg.
        """
        # robot, ground, sensors
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor  # this is weired but it works

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # import ipdb; ipdb.set_trace()
        # actions = torch.zeros_like(actions)
        self.prev_actions = self.actions.clone()    # record the prev action
        self.actions = self.controller.process_action(actions).clone()
        # assert self.actions.any().item() is False
        # assert self.robot.data.joint_pos.any().item() is False

    def _apply_action(self):
        # forces = self.action_scale * self.joint_gears * self.actions
        # self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)
        self.robot.set_joint_position_target(self.actions, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )
    
    def get_robot_description_vec_json(self, file_name: str = "robot_description_vec.json"):
        # Construct the path to the JSON file
        json_path = os.path.join(
            os.path.dirname(
                os.path.dirname(self.cfg.robot.spawn.usd_path)
            ),
            file_name
        )

        # Open and load the JSON file
        with open(json_path, "r") as f:
            self.robot_description_vec_json = json.load(f)

    def quat_to_matrix(self, quat):
        """
        Convert a quaternion to a 3x3 rotation matrix.
        Args:
            quat (torch.tensor): Quaternion (w, x, y, z) with shape (N, 4).
        Returns:
            torch.tensor: Rotation matrix with shape (N, 3, 3).
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Compute the rotation matrix elements
        tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
        twx, twy, twz = tx * w, ty * w, tz * w
        txx, txy, txz = tx * x, tx * y, tx * z
        tyy, tyz, tzz = ty * y, ty * z, tz * z

        rotation_matrix = torch.stack([
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy)
        ], dim=-1).view(-1, 3, 3)

        return rotation_matrix

    def get_name_to_description_vector(self):
        # TODO: Check if this function is robot-dependent, particularly the magic numbers
        name_to_description_vector = {}

        # Root (trunk) position and orientation
        trunk_position_global = self.robot.data.root_pos_w  # Root position
        trunk_orientation_quat = self.robot.data.root_quat_w  # Root orientation (quaternion)

        # Convert trunk orientation to rotation matrix
        trunk_rotation_matrix = self.quat_to_matrix(trunk_orientation_quat).transpose(1, 2)

        # TODO: Implement the robot dimension with geometry bounding box and movable geometry; See below Temporary TODO
        # Get the robot_dimensions from bbox in robot_description_vec.json file
        bbox_corner_0, bbox_corner_1 = torch.tensor(self.robot_description_vec_json['bbox'][0], device=self.sim.device), torch.tensor(self.robot_description_vec_json['bbox'][1], device=self.sim.device)
        robot_dimensions = torch.abs(bbox_corner_0 - bbox_corner_1)
        robot_bbox_center = (bbox_corner_0 + bbox_corner_1) / 2
        self.robot_dimensions = robot_dimensions.unsqueeze(0).repeat(self.num_envs, 1)

        # Calculate robot width, length and height
        num_geom = self.robot.data.body_pos_w.shape[1]
        relative_geom_positions = self.robot.data.body_pos_w - trunk_position_global.unsqueeze(1).repeat(1, num_geom, 1)
        for i in range(len(relative_geom_positions[1])):
            relative_geom_positions[:, i] = torch.matmul(trunk_rotation_matrix.transpose(1,2), relative_geom_positions[:, i].unsqueeze(2)).squeeze(2)

        # Researved for relative_foot_position_normalized
        min_x,_ = torch.min(relative_geom_positions[:,:,0], dim=1)
        min_y,_ = torch.min(relative_geom_positions[:,:,1], dim=1)
        min_z,_ = torch.min(relative_geom_positions[:,:,2], dim=1)
        max_x,_ = torch.max(relative_geom_positions[:,:,0], dim=1)
        max_y,_ = torch.max(relative_geom_positions[:,:,1], dim=1)
        max_z,_ = torch.max(relative_geom_positions[:,:,2], dim=1)
        mins = torch.stack((min_x, min_y, min_z), dim=1)

        # self.robot_length = max_x - min_x
        # self.robot_width = max_y - min_y
        # self.robot_height = max_z - min_z
        # self.robot_dimensions = torch.stack((self.robot_length, self.robot_width, self.robot_height), dim=1)

        self.gains_and_action_scaling_factor = torch.tensor([0, 0, self.cfg.action_scale], device=self.sim.device)
        self.mass = torch.sum(self.robot.data.default_mass, dim=1).unsqueeze(1).to(self.sim.device)

        # Compute normalized joint positions and axes
        for i, joint_name in enumerate(self.joint_names):
            relative_joint_position = torch.tensor(self.robot_description_vec_json['joint_info'][joint_name]['position'], device=self.sim.device).unsqueeze(0).repeat(self.num_envs, 1)
            # relative_joint_position = torch.matmul(trunk_rotation_matrix, relative_joint_position.unsqueeze(-1)).squeeze(-1)
            relative_joint_position -= robot_bbox_center        # noisy center estimate as an approximation 
            relative_joint_position_normalized = (relative_joint_position - mins) / (self.robot_dimensions + 1e-8)  # note the mins are based on links

            # joint_position_global = self.robot.data.joint_pos[:, i].unsqueeze(1)  # Assuming joint positions are indexed similarly to body names
            # joint_position_default = self.robot.data.default_joint_pos[:, i].unsqueeze(1)
            # relative_joint_position_global = joint_position_global - joint_position_default
            # relative_joint_position_normalized = relative_joint_position_global / (self.robot.data.soft_joint_pos_limits[:, i, 0] - self.robot.data.soft_joint_pos_limits[:, i, 1]).unsqueeze(1)

            # TODO: Get the value for relative_joint_axis_local
            relative_joint_axis_local = torch.tensor(self.robot_description_vec_json['joint_info'][joint_name]['axis'], device=self.sim.device).unsqueeze(0).repeat(self.num_envs, 1)

            # joint_axis_global = self.robot.data.body_quat_w[i]  # Replace with actual axis retrieval if available
            # relative_joint_axis_local = torch.matmul(trunk_rotation_matrix, joint_axis_global)

            # Append joint description vector
            name_to_description_vector[joint_name] = torch.cat([
                (relative_joint_position_normalized / 0.5) - 1.0,
                relative_joint_axis_local,
                torch.tensor([self.joint_nr_direct_child_joints[i]], device=self.sim.device).repeat((self.num_envs, 1)),
                torch.tensor([0 / 4.6], device=self.sim.device).repeat((self.num_envs, 1)),
                torch.tensor([(2 / 500.0) - 1.0], device=self.sim.device).repeat((self.num_envs, 1)),
                torch.tensor([(1.75 / 17.5) - 1.0], device=self.sim.device).repeat((self.num_envs, 1)),
                torch.tensor([(0.1 / 5.0) - 1.0], device=self.sim.device).repeat((self.num_envs, 1)),
                torch.tensor([(0.01 / 0.1) - 1.0], device=self.sim.device).repeat((self.num_envs, 1)),
                torch.tensor([(0 / 15.0) - 1.0], device=self.sim.device).repeat((self.num_envs, 1)),
                torch.tensor([(0.03 / 0.6) - 1.0], device=self.sim.device).repeat((self.num_envs, 1)),
                (torch.tensor([-2, 2], device=self.sim.device) / 4.6).repeat((self.num_envs, 1)),
                ((self.gains_and_action_scaling_factor / torch.tensor([50.0, 1.0, 0.4], device=self.sim.device)) - 1.0).repeat((self.num_envs, 1)),
                (self.mass / 85.0) - 1.0,
                (self.robot_dimensions / 1.0) - 1.0,
            ], dim=1)

        # Compute normalized foot positions
        for i, foot_name in enumerate(self.foot_names):
            if "foot" not in foot_name:  # Skip non-foot bodies
                continue
            
            foot_position_global = self.robot.data.body_pos_w[:, i]  # Foot global position
            relative_foot_position_global = foot_position_global - trunk_position_global
            relative_foot_position_local = torch.matmul(trunk_rotation_matrix, relative_foot_position_global.unsqueeze(-1)).squeeze(-1)
            # TODO: now we are using a hack of adding epsilon to robot_dimensions, but remember that we are not using link to compute size
            relative_foot_position_normalized = (relative_foot_position_local - mins) / (self.robot_dimensions + 1e-8)

            # Append foot description vector
            name_to_description_vector[foot_name] = torch.cat([
                (relative_foot_position_normalized / 0.5) - 1.0,
                ((self.gains_and_action_scaling_factor / torch.tensor([50.0, 1.0, 0.4], device=self.sim.device)) - 1.0).repeat((self.num_envs, 1)),
                ((self.mass / 85.0) - 1.0),
                (self.robot_dimensions / 1.0) - 1.0,
            ], dim=1)

        self.dynamic_joint_description_size = name_to_description_vector[self.joint_names[0]].shape[1]
        for foot_name in self.foot_names:
            if "foot" in foot_name:
                self.dynamic_foot_description_size = name_to_description_vector[foot_name].shape[1]
                break

        return name_to_description_vector
    
    def get_initial_observation(self, multi_robot_max_observation_size):
        # Dynamic observations
        dynamic_joint_observations = torch.empty((self.num_envs, 0), device=self.sim.device)
        for i, joint_name in enumerate(self.joint_names):
            desc_vector = self.name_to_description_vector[joint_name]
            joint_pos_rel = (self.robot.data.joint_pos[:, i] - self.robot.data.default_joint_pos[:, i]).unsqueeze(1)
            joint_vel_rel = (self.robot.data.joint_vel[:, i] - self.robot.data.default_joint_vel[:, i]).unsqueeze(1)
            action = self.actions[:, i].unsqueeze(1)

            # Concatenate all the current observations into one tensor
            current_observation = torch.cat((desc_vector, joint_pos_rel, joint_vel_rel, action), dim=1)

            # Concatenate along dimension 0 for subsequent observations
            dynamic_joint_observations = torch.cat((dynamic_joint_observations, current_observation), dim=1)

        # Get the foot contacts with ground
        undesired_contacts = self.get_contacts_without_sum(self.reward_cfgs['feet_ground_contact_cfg'],
                                                         threshold=1.0)
        # Get the time since last foot contact with ground
        feet_air_time = self.get_feet_air_time(
            self.reward_cfgs['feet_ground_contact_cfg'],
            threshold_min=0.2, threshold_max=0.5
        )

        # from training.environments.unitree_go1.info import touchdown_feet
        # Dynamic observations
        dynamic_foot_observations = torch.empty((self.num_envs, 0), device=self.sim.device)  # Start with None for the first iteration
        foot_index = 0
        for i, foot_name in enumerate(self.foot_names):
            if "foot" in foot_name:
                # Concatenate all the current observations into one tensor
                current_observation = torch.cat((self.name_to_description_vector[foot_name], undesired_contacts[:, foot_index].unsqueeze(1), feet_air_time[:, foot_index].unsqueeze(1)), dim=1)
                foot_index += 1
                # Concatenate along dimension 0 for subsequent observations
                dynamic_foot_observations = torch.cat((dynamic_foot_observations, current_observation), dim=1)

        # General observations
        trunk_linear_velocity = self.robot.data.root_lin_vel_b
        trunk_angular_velocity = self.robot.data.root_ang_vel_b
        self.target_x_vel, self.target_y_vel, self.target_yaw_vel = self.command_generator.get_next_command()
        goal_velocity = torch.cat((self.target_x_vel, self.target_y_vel, self.target_yaw_vel), dim=1)
        projected_gravity_vector = self.robot.data.projected_gravity_b
        height = self.robot.data.root_pos_w[:,2].unsqueeze(1)

        # General robot context
        gains_and_action_scaling_factor = ((self.gains_and_action_scaling_factor / torch.tensor([100.0 / 2, 2.0 / 2, 0.8 / 2], device=self.sim.device)) - 1.0).repeat((self.num_envs, 1))
        mass = (self.mass / (170.0 / 2)) - 1.0
        robot_dimensions = ((self.robot_dimensions / (2.0 / 2)) - 1.0)

        # Padding
        padding = torch.empty((self.num_envs, 0), device=self.sim.device)
        # if multi_robot_max_observation_size != -1:
        #     padding = torch.zeros(self.missing_nr_of_observations, dtype=torch.float32)

        # Temporary value for expanding scalar to env_num tensor

        observation = torch.cat([
            dynamic_joint_observations,
            dynamic_foot_observations,
            trunk_linear_velocity,
            trunk_angular_velocity,
            goal_velocity,
            projected_gravity_vector,
            height,
            gains_and_action_scaling_factor,
            mass,
            robot_dimensions,
            padding,
        ], dim=1)

        return observation
    
    def get_observation_space(self, multi_robot_max_observation_size):
        observation_names = []

        # Dynamic observations
        self.nr_dynamic_joint_observations = len(self.joint_names)
        self.single_dynamic_joint_observation_length = self.dynamic_joint_description_size + 3
        self.dynamic_joint_observation_length = self.single_dynamic_joint_observation_length * self.nr_dynamic_joint_observations
        for joint_name in self.joint_names:
            observation_names.extend([joint_name + "_description_" + str(i) for i in range(self.dynamic_joint_description_size)])
            observation_names.extend([
                joint_name + "_position", joint_name + "_velocity", joint_name + "_previous_action",
            ])

        self.nr_dynamic_foot_observations = len([foot_name for foot_name in self.foot_names if "foot" in foot_name])
        self.single_dynamic_foot_observation_length = self.dynamic_foot_description_size + 2
        self.dynamic_foot_observation_length = self.single_dynamic_foot_observation_length * self.nr_dynamic_foot_observations
        for foot_name in self.foot_names:
            if "foot" in foot_name:
                observation_names.extend([foot_name + "_description_" + str(i) for i in range(self.dynamic_foot_description_size)])
                observation_names.extend([
                    foot_name + "_ground_contact", foot_name + "_cycles_since_last_ground_contact",
                ])

        # General observations
        observation_names.extend([
            "trunk_x_velocity", "trunk_y_velocity", "trunk_z_velocity",
            "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
        ])

        observation_names.extend(["goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity"])
        observation_names.extend(["projected_gravity_x", "projected_gravity_y", "projected_gravity_z"])
        observation_names.append("height_0")

        # General robot context
        observation_names.extend(["p_gain", "d_gain", "action_scaling_factor"])
        observation_names.append("mass")
        observation_names.extend(["robot_length", "robot_width", "robot_height"])

        self.missing_nr_of_observations = 0
        if multi_robot_max_observation_size != -1:
            self.missing_nr_of_observations = multi_robot_max_observation_size - len(observation_names)
            observation_names.extend(["padding_" + str(i) for i in range(self.missing_nr_of_observations)])

        name_to_idx = {name: idx for idx, name in enumerate(observation_names)}

        self.one_policy_observation_length = len(name_to_idx)

        return name_to_idx

    def _get_observations(self) -> dict:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b
        self.target_x_vel, self.target_y_vel, self.target_yaw_vel = self.command_generator.get_next_command()
        joint_pos_rel = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel_rel = self.robot.data.joint_vel - self.robot.data.default_joint_vel
        actions = self.actions  # actions at prev step
        # base_lin_vel, base_ang_vel, joint_pos_rel, joint_vel_rel could explode
        # import ipdb; ipdb.set_trace()
        normal_obs = torch.cat(
            [base_lin_vel, base_ang_vel, projected_gravity_b, self.target_x_vel, self.target_y_vel,
             self.target_yaw_vel, joint_pos_rel, joint_vel_rel, actions], dim=1
        )

        ## Define the one policy observations below
        # Copy the initial observation for update
        urma_obs = self.initial_observation.clone()

        # Get the foot contacts with ground
        undesired_contacts = self.get_contacts_without_sum(self.reward_cfgs['feet_ground_contact_cfg'],
                                                         threshold=1.0)
        
        # Get the time since last foot contact with ground
        feet_air_time = self.get_feet_air_time(
            self.reward_cfgs['feet_ground_contact_cfg'],
            threshold_min=0.2, threshold_max=0.5
        )

        # Update observations every step
        urma_obs[:, self.joint_positions_update_obs_idx] = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        urma_obs[:, self.joint_velocities_update_obs_idx] = self.robot.data.joint_vel - self.robot.data.default_joint_vel
        urma_obs[:, self.joint_previous_actions_update_obs_idx] = self.actions
        # note for the undesired_contacts dimension
        urma_obs[:, self.foot_ground_contact_update_obs_idx] = undesired_contacts.type(torch.float32)
        urma_obs[:, self.foot_time_since_last_ground_contact_update_obs_idx] = feet_air_time
        urma_obs[:, self.trunk_linear_vel_update_obs_idx] = self.robot.data.root_lin_vel_b
        urma_obs[:, self.trunk_angular_vel_update_obs_idx] = self.robot.data.root_ang_vel_b
        urma_obs[:, self.goal_velocity_update_obs_idx] = torch.cat((self.target_x_vel, self.target_y_vel, self.target_yaw_vel), dim=1)
        urma_obs[:, self.projected_gravity_update_obs_idx] = self.robot.data.projected_gravity_b
        urma_obs[:, self.height_update_obs_idx] = self.robot.data.root_pos_w[:,2].unsqueeze(1)

        # Add noise
        # observation = self.observation_noise_function.modify_observation(observation)

        # # Dropout
        # observation = self.observation_dropout_function.modify_observation(observation)
        # import random
        # if random.random() > 0.995:
        #     import ipdb; ipdb.set_trace()

        # # Normalize and clip
        # observation[:, self.joint_positions_update_obs_idx] /= 4.6      # max is only 1.06? mean 0.0
        # observation[:, self.joint_velocities_update_obs_idx] /= 35.0    # range from -3 to 3?   
        # observation[:, self.joint_previous_actions_update_obs_idx] /= 10.0      # looks reasonable; range from -10 to 10
        # observation[:, self.foot_ground_contact_update_obs_idx] = (observation[:, self.foot_ground_contact_update_obs_idx] / 0.5) - 1.0 # taking either 1 or 0 
        # observation[:, self.foot_time_since_last_ground_contact_update_obs_idx] = torch.clip((observation[:, self.foot_time_since_last_ground_contact_update_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)    # range from -0.3 to 0.3?
        # observation[:, self.trunk_linear_vel_update_obs_idx] = torch.clip(observation[:, self.trunk_linear_vel_update_obs_idx] / 10.0, -1.0, 1.0)  # range from -1.x to 1? 
        # observation[:, self.trunk_angular_vel_update_obs_idx] = torch.clip(observation[:, self.trunk_angular_vel_update_obs_idx] / 50.0, -1.0, 1.0)  # range from -3 to 3
        # observation[:, self.height_update_obs_idx] = torch.clip((observation[:, self.height_update_obs_idx] / (2*self.robot_dimensions[:, 2].unsqueeze(1) / 2)) - 1.0, -1.0, 1.0)  # the quotient ranges from 0 to 1.x? 

        urma_obs[:, self.joint_positions_update_obs_idx] /= 3.14      # max is only 1.06? mean 0.0
        urma_obs[:, self.joint_velocities_update_obs_idx] /= 3    # range from -3 to 3?
        urma_obs[:, self.joint_previous_actions_update_obs_idx] /= 10.0      # looks reasonable; range from -10 to 10
        urma_obs[:, self.foot_ground_contact_update_obs_idx] = (urma_obs[:, self.foot_ground_contact_update_obs_idx] / 0.5) - 1.0 # taking either 1 or 0
        urma_obs[:, self.foot_time_since_last_ground_contact_update_obs_idx] = torch.clip((urma_obs[:, self.foot_time_since_last_ground_contact_update_obs_idx] / 1) - 1.0, -1.0, 1.0)    # range from -0.3 to 0.3?
        urma_obs[:, self.trunk_linear_vel_update_obs_idx] = torch.clip(urma_obs[:, self.trunk_linear_vel_update_obs_idx] / 1, -2.0, 2.0)  # range from -1.x to 1?
        urma_obs[:, self.trunk_angular_vel_update_obs_idx] = torch.clip(urma_obs[:, self.trunk_angular_vel_update_obs_idx] / 3.14, -3.0, 3.0)  # range from -3 to 3
        urma_obs[:, self.height_update_obs_idx] = torch.clip((urma_obs[:, self.height_update_obs_idx] / (2*self.robot_dimensions[:, 2].unsqueeze(1) / 2)) - 1.0, -2.0, 2.0)  # the quotient ranges from 0 to 1.x?

        ## Define the one policy observations above

        # for i, x in enumerate([base_lin_vel, base_ang_vel, projected_gravity_b, self.target_x_vel, self.target_y_vel,
        #      self.target_yaw_vel, joint_pos_rel, joint_vel_rel, actions]):
        #     print(i, x.max())

        # obs = torch.cat(
        #     (
        #         self.target_x_vel,
        #         self.target_y_vel,
        #         self.target_yaw_vel,
        #         lin_vel,
        #         ang_vel,
        #         self.torso_position[:, 2].view(-1, 1),
        #         self.vel_loc,
        #         self.angvel_loc * self.cfg.angular_velocity_scale,
        #         normalize_angle(self.yaw).unsqueeze(-1),
        #         normalize_angle(self.roll).unsqueeze(-1),
        #         normalize_angle(self.angle_to_target).unsqueeze(-1),
        #         self.up_proj.unsqueeze(-1),
        #         self.heading_proj.unsqueeze(-1),
        #         self.dof_pos_scaled,
        #         self.dof_vel * self.cfg.dof_vel_scale,
        #         self.actions,
        #     ),
        #     dim=-1,
        # )

        observations = {"policy": normal_obs, "one_policy": urma_obs}

        return observations

    def get_feet_air_time_reward(self, sensor_cfg, threshold_min, threshold_max):
        """
        Encourage taking large strides by rewarding long air time
        """
        contact_sensor: ContactSensor = self.scene.sensors[sensor_cfg.name]
        # check if contact is made in the past self.step_dt second
        first_contact = contact_sensor.compute_first_contact(self.step_dt)[:, sensor_cfg.body_ids]
        # check time spent in the air before last contact (in sec)
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        # negative reward for small steps to discourage the behavior
        # note that we only compute reward for legs that make contact recently
        # so that we don't repeatedly compute reward for every step
        # for example, if a leg keeps staying in the air, this long air time will only
        # be penalized after it gets in contact with ground next time
        air_time = (last_air_time - threshold_min) * first_contact
        # no reward for large steps
        air_time = torch.clamp(air_time, max=threshold_max-threshold_min)
        reward = torch.sum(air_time, dim=1)
        # no reward for zero command
        command = torch.cat([self.target_x_vel, self.target_y_vel, self.target_yaw_vel], dim=1)
        reward *= torch.norm(command[:, :2], dim=1) > 0.1
        return reward

    def get_feet_air_time(self, sensor_cfg, threshold_min, threshold_max):
        contact_sensor: ContactSensor = self.scene.sensors[sensor_cfg.name]
        # check if contact is made in the past self.step_dt second
        first_contact = contact_sensor.compute_first_contact(self.step_dt)[:, sensor_cfg.body_ids]
        # check time spent in the air before last contact (in sec)
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        # negative reward for small steps to discourage the behavior
        # note that we only compute reward for legs that make contact recently
        # so that we don't repeatedly compute reward for every step
        # for example, if a leg keeps staying in the air, this long air time will only
        # be penalized after it gets in contact with ground next time
        air_time = (last_air_time - threshold_min) * first_contact
        # no reward for large steps
        air_time = torch.clamp(air_time, max=threshold_max-threshold_min)
        return air_time
    
    def get_feet_slide_reward(self, sensor_cfg, asset_cfg):
        """
        Penalize feet sliding, where sliding is defined by the leg having contact AND velocity
        This means that it does not penalize contact that has no velocity, e.g., in-place rotation
        """
        contact_sensor: ContactSensor = self.scene.sensors[sensor_cfg.name]
        contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        asset = self.scene[asset_cfg.name]
        # sliding is characterized by having contact AND the body part has velocity
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
        return reward

    def get_undesired_contacts(self, sensor_cfg, threshold) -> torch.Tensor:
        """Penalize undesired contacts as the number of violations that are above a threshold."""
        # # create sensor cfg
        # sensor_cfg = SceneEntityCfg(sensor_name, body_names=body_names)
        # sensor_cfg.resolve(self.scene)
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = self.scene.sensors[sensor_cfg.name]
        # check if contact force is above threshold
        net_contact_forces = contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        # sum over contacts for each environment
        #TODO: check if the returned data type (Long) makes sense for this reward
        return torch.sum(is_contact, dim=1)
    
    def get_contacts_without_sum(self, sensor_cfg, threshold) -> torch.Tensor:
        """Penalize undesired contacts as the number of violations that are above a threshold."""
        # # create sensor cfg
        # sensor_cfg = SceneEntityCfg(sensor_name, body_names=body_names)
        # sensor_cfg.resolve(self.scene)
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = self.scene.sensors[sensor_cfg.name]
        # check if contact force is above threshold
        net_contact_forces = contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        # sum over contacts for each environment
        #TODO: check if the returned data type (Long) makes sense for this reward
        return is_contact

    def get_joint_deviation_l1(self, asset_cfg) -> torch.Tensor:
        """Penalize joint positions that deviate from the default one."""
        # # create asset cfg
        # asset_cfg = SceneEntityCfg(asset_name, joint_names=joint_names)
        # asset_cfg.resolve(self.scene)
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self.scene[asset_cfg.name]
        # compute out of limits constraints
        angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        return torch.sum(torch.abs(angle), dim=1)

    def get_vertical_deviation_l1(self):
        """Computes the deviation from vertical direction using L1 metric"""
        return torch.abs(1.0 - self.up_proj)

    # @torch.jit.script
    def _get_rewards(self) -> torch.Tensor:
        # return torch.zeros(4096).to(self.device)
        # start_block = time.time()
        # init_time = start_block

        # get linear and angular vel in world frame
        lin_vel, ang_vel = self.robot.data.root_lin_vel_b, self.robot.data.root_ang_vel_b

        # task reward, check velocity
        track_lin_vel_xy_exp = track_vel_exp(lin_vel[:, :2],
                                             torch.cat([self.target_x_vel, self.target_y_vel], dim=1))
        track_ang_vel_z_exp = track_vel_exp(lin_vel[:, 1:], self.target_yaw_vel[:, :1])

        # penalties
        # penalize linear velocity in z axis
        lin_vel_z_l2 = lin_vel[:, 2] ** 2
        # penalize angular velocity in xy axis
        ang_vel_xy_l2 = (ang_vel[:, :2] ** 2).sum(1)

        # penalize joint torques
        joint_torques_l2 = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
        # penalize action change rate
        action_rate_l2 = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)

        # compute reward that encourage large strides
        feet_air_time = self.get_feet_air_time_reward(
            self.reward_cfgs['feet_ground_contact_cfg'],
            threshold_min=0.2, threshold_max=0.5
        )

        # compute reward that disencourages sliding
        feet_slide = self.get_feet_slide_reward(self.reward_cfgs['feet_ground_contact_cfg'],
                                                self.reward_cfgs['feet_ground_asset_cfg'])

        # undesired contacts
        undesired_contacts = self.get_undesired_contacts(self.reward_cfgs['undesired_contact_cfg'],
                                                         threshold=1.0)

        # compute deviation for hip
        joint_deviation_hip = self.get_joint_deviation_l1(self.reward_cfgs['joint_hip_cfg'])

        # compute deviation for knee
        joint_deviation_knee = self.get_joint_deviation_l1(self.reward_cfgs['joint_knee_cfg'])

        # compute deviation from the vertical direction
        vertical_reward = self.get_vertical_deviation_l1()

        # # self.cfg.rewards
        # total_reward = compute_rewards(
        #     self.actions,
        #     self.reset_terminated,
        #     self.cfg.up_weight,
        #     self.cfg.heading_weight,
        #     self.heading_proj,
        #     self.up_proj,
        #     self.dof_vel,
        #     self.dof_pos_scaled,
        #     self.potentials,
        #     self.prev_potentials,
        #     self.cfg.actions_cost_scale,
        #     self.cfg.energy_cost_scale,
        #     self.cfg.dof_vel_scale,
        #     self.cfg.death_cost,
        #     self.cfg.alive_reward_scale,
        #     self.motor_effort_ratio,
        # )

        self.reward_dict = {
            'track_lin_vel_xy_exp': track_lin_vel_xy_exp * 1.0 * 2,
            'track_ang_vel_z_exp': track_ang_vel_z_exp * 0.5 * 2,
            'lin_vel_z_l2': lin_vel_z_l2 * -2.0,
            'ang_vel_xy_l2': ang_vel_xy_l2 * -0.05,
            'joint_torques_l2': joint_torques_l2 * -1e-5,
            'action_rate_l2': action_rate_l2 * -0.01,
            'feet_air_time': feet_air_time * 2.0,
            'feet_slide': feet_slide * -0.25,
            'undesired_contacts': undesired_contacts * -0.1,
            'joint_deviation_hip': joint_deviation_hip * -0.1 * 5 * 0.2,
            'joint_deviation_knee': joint_deviation_knee * -0.01 * 0.2,
            'vertical_reward': vertical_reward * -0.1
        }
        total_reward = sum(self.reward_dict.values())

        # print(f'total time: {time.time() - init_time}')
        # print('reward', {k: v.mean() for k, v in self.reward_dict.items()})
        # import ipdb; ipdb.set_trace()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # died = self.torso_position[:, 2] < self.cfg.termination_height
        # judge based on illegal contact
        illegal_contact_cfg = self.reward_cfgs['illegal_contact_cfg']
        contact_sensor = self.scene.sensors[illegal_contact_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, illegal_contact_cfg.body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

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

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()


# # @torch.jit.script
# def compute_rewards(
#     actions: torch.Tensor,
#     reset_terminated: torch.Tensor,
#     up_weight: float,
#     heading_weight: float,
#     heading_proj: torch.Tensor,
#     up_proj: torch.Tensor,
#     dof_vel: torch.Tensor,
#     dof_pos_scaled: torch.Tensor,
#     potentials: torch.Tensor,
#     prev_potentials: torch.Tensor,
#     actions_cost_scale: float,
#     energy_cost_scale: float,
#     dof_vel_scale: float,
#     death_cost: float,
#     alive_reward_scale: float,
#     motor_effort_ratio: torch.Tensor,
# ):
#     heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
#     heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)
#
#     # aligning up axis of robot and environment
#     up_reward = torch.zeros_like(heading_reward)
#     up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)
#
#     # energy penalty for movement
#     actions_cost = torch.sum(actions**2, dim=-1)
#     electricity_cost = torch.sum(
#         torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
#         dim=-1,
#     )
#
#     # dof at limit cost
#     dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)
#
#     # reward for duration of staying alive
#     alive_reward = torch.ones_like(potentials) * alive_reward_scale
#     progress_reward = potentials - prev_potentials
#
#     total_reward = (
#         progress_reward
#         + alive_reward
#         + up_reward
#         + heading_reward
#         - actions_cost_scale * actions_cost
#         - energy_cost_scale * electricity_cost
#         - dof_at_limit_cost
#     )
#
#     # adjust reward for fallen agents
#     total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
#     return total_reward


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
