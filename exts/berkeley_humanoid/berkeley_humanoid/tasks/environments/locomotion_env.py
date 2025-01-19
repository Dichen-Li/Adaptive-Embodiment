from __future__ import annotations
import torch
import math

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.sensors import ContactSensor
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from .utils import quat_to_matrix
import os
import json


class LocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg


    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_dt = self.cfg.action_dt

        self.all_bodies_cfg = self.cfg.all_bodies_cfg
        self.all_bodies_cfg.resolve(self.scene)
        self.all_joints_cfg = self.cfg.all_joints_cfg
        self.all_joints_cfg.resolve(self.scene)
        self.trunk_cfg = self.cfg.trunk_cfg
        self.trunk_cfg.resolve(self.scene)
        self.trunk_contact_cfg = self.cfg.trunk_contact_cfg
        self.trunk_contact_cfg.resolve(self.scene)
        self.feet_contact_cfg = self.cfg.feet_contact_cfg
        self.feet_contact_cfg.resolve(self.scene)

        self.nr_joints = self.robot.data.default_joint_pos.shape[1]
        self.nr_feet = self.cfg.nr_feet

        self.joint_nominal_positions = self.robot.data.default_joint_pos
        self.action_scaling_factor = self.cfg.action_scaling_factor
        if 'all' in self.robot.actuators:
            robot_actuators = self.robot.actuators["all"] # this should use the same order as self.robot.data.joint_names
            self.joint_max_velocity = robot_actuators.velocity_limit
            self.joint_max_torque = robot_actuators.effort_limit
            self.p_gains = robot_actuators.stiffness
            self.d_gains = robot_actuators.damping
        elif 'base_legs' in self.robot.actuators:
            robot_actuators = self.robot.actuators["base_legs"]
            self.joint_max_velocity = robot_actuators.velocity_limit
            self.joint_max_torque = robot_actuators.effort_limit
            self.p_gains = robot_actuators.stiffness
            self.d_gains = robot_actuators.damping
        else:
            raise NotImplementedError()

        self.step_sampling_probability = self.cfg.step_sampling_probability

        self.goal_velocities = torch.rand((self.num_envs, 3), device=self.sim.device).uniform_(-1.0, 1.0)
        self.previous_actions = torch.zeros((self.num_envs, self.nr_joints), dtype=torch.float32, device=self.sim.device)
        self.previous_feet_air_times = torch.zeros((self.num_envs, self.nr_feet), dtype=torch.float32, device=self.sim.device)

        self.reward_curriculum_steps = self.cfg.reward_curriculum_steps
        self.tracking_xy_velocity_command_coeff = self.cfg.tracking_xy_velocity_command_coeff
        self.tracking_yaw_velocity_command_coeff = self.cfg.tracking_yaw_velocity_command_coeff
        self.z_velocity_coeff = self.cfg.z_velocity_coeff
        self.pitch_roll_vel_coeff = self.cfg.pitch_roll_vel_coeff
        self.pitch_roll_pos_coeff = self.cfg.pitch_roll_pos_coeff
        self.actuator_joint_nominal_diff_coeff = self.cfg.actuator_joint_nominal_diff_coeff
        actuator_joint_nominal_diff_joints_cfg = self.cfg.actuator_joint_nominal_diff_joints_cfg
        if actuator_joint_nominal_diff_joints_cfg is not None:
            actuator_joint_nominal_diff_joints_cfg.resolve(self.scene)
            self.actuator_joint_nominal_diff_joints = actuator_joint_nominal_diff_joints_cfg.joint_ids
        else:
            # dummy to first joint to avoid tensor indexing error -> in that case make sure the coeff is 0.0
            self.actuator_joint_nominal_diff_joints = [0,]  
            self.actuator_joint_nominal_diff_coeff = 0.0
        self.joint_position_limit_coeff = self.cfg.joint_position_limit_coeff
        self.joint_acceleration_coeff = self.cfg.joint_acceleration_coeff
        self.joint_torque_coeff = self.cfg.joint_torque_coeff
        self.action_rate_coeff = self.cfg.action_rate_coeff
        self.base_height_coeff = self.cfg.base_height_coeff
        self.air_time_coeff = self.cfg.air_time_coeff
        self.symmetry_air_coeff = self.cfg.symmetry_air_coeff
        self.feet_symmetry_pairs = torch.tensor(self.cfg.feet_symmetry_pairs, dtype=torch.int32, device=self.sim.device)
        self.feet_y_distance_coeff = self.cfg.feet_y_distance_coeff
        self.calculated_feet_y_distance_target = False
        self.domain_randomization_curriculum_steps = self.cfg.domain_randomization_curriculum_steps

        self.max_nr_action_delay_steps = self.cfg.max_nr_action_delay_steps
        self.mixed_action_delay_chance = self.cfg.mixed_action_delay_chance
        self.action_current_mixed = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.sim.device)
        self.action_history = torch.zeros((self.num_envs, self.max_nr_action_delay_steps + 1, self.nr_joints), dtype=torch.float32, device=self.sim.device)

        self.motor_strength_min = self.cfg.motor_strength_min
        self.motor_strength_max = self.cfg.motor_strength_max
        self.p_gain_factor_min = self.cfg.p_gain_factor_min
        self.p_gain_factor_max = self.cfg.p_gain_factor_max
        self.d_gain_factor_min = self.cfg.d_gain_factor_min
        self.d_gain_factor_max = self.cfg.d_gain_factor_max
        self.p_law_position_offset_min = self.cfg.p_law_position_offset_min
        self.p_law_position_offset_max = self.cfg.p_law_position_offset_max
        self.extrinsic_motor_strength = torch.ones((self.num_envs, self.nr_joints), dtype=torch.float32, device=self.sim.device)
        self.extrinsic_p_gain_factor = torch.ones((self.num_envs, self.nr_joints), dtype=torch.float32, device=self.sim.device)
        self.extrinsic_d_gain_factor = torch.ones((self.num_envs, self.nr_joints), dtype=torch.float32, device=self.sim.device)
        self.extrinsic_position_offset = torch.zeros((self.num_envs, self.nr_joints), dtype=torch.float32, device=self.sim.device)

        self.initial_state_roll_angle_factor = self.cfg.initial_state_roll_angle_factor
        self.initial_state_pitch_angle_factor = self.cfg.initial_state_pitch_angle_factor
        self.initial_state_yaw_angle_factor = self.cfg.initial_state_yaw_angle_factor
        self.initial_state_joint_nominal_position_factor = self.cfg.initial_state_joint_nominal_position_factor
        self.initial_state_joint_velocity_factor = self.cfg.initial_state_joint_velocity_factor
        self.initial_state_joint_velocity_clip = self.cfg.initial_state_joint_velocity_clip
        self.initial_state_max_linear_velocity = self.cfg.initial_state_max_linear_velocity
        self.initial_state_max_angular_velocity = self.cfg.initial_state_max_angular_velocity

        self.joint_position_noise = self.cfg.joint_position_noise
        self.joint_velocity_noise = self.cfg.joint_velocity_noise
        self.trunk_angular_velocity_noise = self.cfg.trunk_angular_velocity_noise
        self.ground_contact_noise_chance = self.cfg.ground_contact_noise_chance
        self.contact_time_noise_chance = self.cfg.contact_time_noise_chance
        self.contact_time_noise_factor = self.cfg.contact_time_noise_factor
        self.gravity_vector_noise = self.cfg.gravity_vector_noise

        self.joint_and_feet_dropout_chance = self.cfg.joint_and_feet_dropout_chance

        self.static_friction_min = self.cfg.static_friction_min
        self.static_friction_max = self.cfg.static_friction_max
        self.dynamic_friction_min = self.cfg.dynamic_friction_min
        self.dynamic_friction_max = self.cfg.dynamic_friction_max
        self.restitution_min = self.cfg.restitution_min
        self.restitution_max = self.cfg.restitution_max
        self.added_trunk_mass_min = self.cfg.added_trunk_mass_min
        self.added_trunk_mass_max = self.cfg.added_trunk_mass_max
        self.added_gravity_min = self.cfg.added_gravity_min
        self.added_gravity_max = self.cfg.added_gravity_max
        self.joint_friction_min = self.cfg.joint_friction_min
        self.joint_friction_max = self.cfg.joint_friction_max
        self.joint_armature_min = self.cfg.joint_armature_min
        self.joint_armature_max = self.cfg.joint_armature_max
        event_term_config = mdp.EventTermCfg()
        event_term_config.params = {
            "env": self,
            "static_friction_range": (self.static_friction_min, self.static_friction_max),
            "dynamic_friction_range": (self.dynamic_friction_min, self.dynamic_friction_max),
            "restitution_range": (self.restitution_min, self.restitution_max),
            "num_buckets": 64,
            "asset_cfg": self.all_bodies_cfg,
        }
        self.randomize_rigid_body_material = mdp.randomize_rigid_body_material(event_term_config, self)

        self.perturb_velocity_x_min = self.cfg.perturb_velocity_x_min
        self.perturb_velocity_x_max = self.cfg.perturb_velocity_x_max
        self.perturb_velocity_y_min = self.cfg.perturb_velocity_y_min
        self.perturb_velocity_y_max = self.cfg.perturb_velocity_y_max
        self.perturb_velocity_z_min = self.cfg.perturb_velocity_z_min
        self.perturb_velocity_z_max = self.cfg.perturb_velocity_z_max
        self.perturb_add_chance = self.cfg.perturb_add_chance
        self.perturb_additive_multiplier = self.cfg.perturb_additive_multiplier

        self.set_observation_indices()

        # initialize URMA-related attributs
        self.urma_initialized = False
        self._setup_urma_obs_params()

        self.handle_domain_randomization(all_envs=True)

    def _setup_urma_obs_params(self):
        """
        Migrated from branch isaac-v2.1. Initializing some attributes needed to generate urma observation.
        """
        self._read_robot_description_vec()

        self.joint_names = self.robot.data.joint_names
        self.foot_names = self.robot.data.body_names

        self.name_to_description_vector = self.get_name_to_description_vector()
        self.initial_observation = self._initialize_urma_observations()
        self.observation_name_to_id = self._get_observation_space()

        # Idx that need to be updated every step
        self.joint_positions_update_obs_idx = [self.observation_name_to_id[joint_name + "_position"] for joint_name in self.joint_names]
        self.joint_velocities_update_obs_idx = [self.observation_name_to_id[joint_name + "_velocity"] for joint_name in self.joint_names]
        self.joint_previous_actions_update_obs_idx = [self.observation_name_to_id[joint_name + "_previous_action"] for joint_name in self.joint_names]
        self.trunk_angular_vel_update_obs_idx = [self.observation_name_to_id["trunk_" + observation_name] for observation_name in ["roll_velocity", "pitch_velocity", "yaw_velocity"]]
        self.goal_velocity_update_obs_idx = [self.observation_name_to_id["goal_" + observation_name] for observation_name in ["x_velocity", "y_velocity", "yaw_velocity"]]
        self.projected_gravity_update_obs_idx = [self.observation_name_to_id["projected_gravity_" + observation_name] for observation_name in ["x", "y", "z"]]

        self.urma_initialized = True
        print("LocomotionEnv init done")

    def _get_observation_space(self):
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

        # General observations
        observation_names.extend([
            "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
        ])

        observation_names.extend(["goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity"])
        observation_names.extend(["projected_gravity_x", "projected_gravity_y", "projected_gravity_z"])

        # General robot context
        observation_names.extend(["p_gain", "d_gain", "action_scaling_factor"])
        observation_names.append("mass")
        observation_names.extend(["robot_length", "robot_width", "robot_height"])

        name_to_idx = {name: idx for idx, name in enumerate(observation_names)}

        self.one_policy_observation_length = len(name_to_idx)

        return name_to_idx

    def _read_robot_description_vec(self, file_name: str = "robot_description_vec.json"):
        # Construct the path to the JSON file
        json_path = os.path.join(
            os.path.dirname(
                os.path.dirname(self.cfg.robot.spawn.usd_path)
            ),
            file_name
        )

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Could not find the robot description JSON file at {json_path}. Did you generate it?"
            )

        # Open and load the JSON file
        with open(json_path, "r") as f:
            self.robot_description_vec_json = json.load(f)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.robot._apply_actuator_model = lambda: None
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

    def get_name_to_description_vector(self):
        name_to_description_vector = {}

        # Root (trunk) position and orientation
        trunk_position_global = self.robot.data.root_pos_w  # Root position
        trunk_orientation_quat = self.robot.data.root_quat_w  # Root orientation (quaternion)

        # Convert trunk orientation to rotation matrix
        trunk_rotation_matrix = quat_to_matrix(trunk_orientation_quat).transpose(1, 2)

        # TODO: Implement the robot dimension with geometry bounding box and movable geometry; See below Temporary TODO
        # Get the robot_dimensions from bbox in robot_description_vec.json file
        bbox_corner_0, bbox_corner_1 = torch.tensor(self.robot_description_vec_json['bbox'][0],
                                                    device=self.sim.device), torch.tensor(
            self.robot_description_vec_json['bbox'][1], device=self.sim.device)
        robot_dimensions = torch.abs(bbox_corner_0 - bbox_corner_1)
        robot_bbox_center = (bbox_corner_0 + bbox_corner_1) / 2
        self.robot_dimensions = robot_dimensions.unsqueeze(0).repeat(self.num_envs, 1)

        # Calculate robot width, length and height
        num_geom = self.robot.data.body_pos_w.shape[1]
        relative_geom_positions = self.robot.data.body_pos_w - trunk_position_global.unsqueeze(1).repeat(1, num_geom, 1)
        for i in range(len(relative_geom_positions[1])):
            relative_geom_positions[:, i] = torch.matmul(trunk_rotation_matrix.transpose(1, 2),
                                                         relative_geom_positions[:, i].unsqueeze(2)).squeeze(2)

        # Researved for relative_foot_position_normalized
        min_x, _ = torch.min(relative_geom_positions[:, :, 0], dim=1)
        min_y, _ = torch.min(relative_geom_positions[:, :, 1], dim=1)
        min_z, _ = torch.min(relative_geom_positions[:, :, 2], dim=1)
        max_x, _ = torch.max(relative_geom_positions[:, :, 0], dim=1)
        max_y, _ = torch.max(relative_geom_positions[:, :, 1], dim=1)
        max_z, _ = torch.max(relative_geom_positions[:, :, 2], dim=1)
        mins = torch.stack((min_x, min_y, min_z), dim=1)

        # self.robot_length = max_x - min_x
        # self.robot_width = max_y - min_y
        # self.robot_height = max_z - min_z
        # self.robot_dimensions = torch.stack((self.robot_length, self.robot_width, self.robot_height), dim=1)

        self.gains_and_action_scaling_factor = torch.tensor([self.p_gains[0, 0], self.d_gains[0, 0],
                                                             self.cfg.action_scaling_factor], device=self.sim.device)
        self.mass = torch.sum(self.robot.data.default_mass, dim=1).unsqueeze(1).to(self.sim.device)

        # Compute normalized joint positions and axes
        for i, joint_name in enumerate(self.joint_names):
            relative_joint_position = torch.tensor(
                self.robot_description_vec_json['joint_info'][joint_name]['position'],
                device=self.sim.device).unsqueeze(0).repeat(self.num_envs, 1)
            # relative_joint_position = torch.matmul(trunk_rotation_matrix, relative_joint_position.unsqueeze(-1)).squeeze(-1)
            relative_joint_position -= robot_bbox_center  # noisy center estimate as an approximation
            relative_joint_position_normalized = (relative_joint_position - mins) / (
                        self.robot_dimensions + 1e-8)  # note the mins are based on links

            # joint_position_global = self.robot.data.joint_pos[:, i].unsqueeze(1)  # Assuming joint positions are indexed similarly to body names
            # joint_position_default = self.robot.data.default_joint_pos[:, i].unsqueeze(1)
            # relative_joint_position_global = joint_position_global - joint_position_default
            # relative_joint_position_normalized = relative_joint_position_global / (self.robot.data.soft_joint_pos_limits[:, i, 0] - self.robot.data.soft_joint_pos_limits[:, i, 1]).unsqueeze(1)

            # TODO: Get the value for relative_joint_axis_local
            relative_joint_axis_local = torch.tensor(self.robot_description_vec_json['joint_info'][joint_name]['axis'],
                                                     device=self.sim.device).unsqueeze(0).repeat(self.num_envs, 1)

            # joint_axis_global = self.robot.data.body_quat_w[i]  # Replace with actual axis retrieval if available
            # relative_joint_axis_local = torch.matmul(trunk_rotation_matrix, joint_axis_global)

            # Append joint description vector
            name_to_description_vector[joint_name] = torch.cat([
                (relative_joint_position_normalized / 0.5) - 1.0,
                relative_joint_axis_local,
                self.joint_nominal_positions[:, i].unsqueeze(1) / 4.6,
                (self.joint_max_torque[:, i].unsqueeze(1) / 500.0) - 1.0,
                (self.joint_max_velocity[:, i].unsqueeze(1) / 17.5) - 1.0,
                self.robot.data.soft_joint_pos_limits[:, i, 0].unsqueeze(1) / 4.6,
                self.robot.data.soft_joint_pos_limits[:, i, 1].unsqueeze(1) / 4.6, 
                ((self.gains_and_action_scaling_factor / torch.tensor([50.0, 1.0, 0.4], device=self.sim.device)) - 1.0).repeat((self.num_envs, 1)),
                (self.mass / 85.0) - 1.0,
                (self.robot_dimensions / 1.0) - 1.0,
            ], dim=1)
        
        self.dynamic_joint_description_size = name_to_description_vector[self.joint_names[0]].shape[1]

        return name_to_description_vector

    def _reset_idx(self, env_ids: torch.Tensor):
        if not self.calculated_feet_y_distance_target:
            self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[env_ids], 0.0 * self.joint_max_velocity[env_ids], None, env_ids)
            feet_indices, _ = self.robot.find_bodies(self.feet_contact_cfg.body_names, True)
            global_feet_pos = self.robot.data.body_pos_w[:, feet_indices]
            local_feet_pos = math_utils.quat_rotate_inverse(self.robot.data.root_quat_w[:, None, :], global_feet_pos - self.robot.data.root_state_w[:, None, :3])
            feet_y_distance = torch.abs(local_feet_pos[:, self.feet_symmetry_pairs[:, 0], 1] - local_feet_pos[:, self.feet_symmetry_pairs[:, 1], 1]).mean(dim=1)
            self.feet_y_distance_target = feet_y_distance.mean().item()
            self.calculated_feet_y_distance_target = True
            self.nominal_trunk_z = self.robot.data.default_root_state[0, 2] - global_feet_pos[:, :, 2].min().item()
        
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        nr_reset_envs = env_ids.shape[0]

        global_step = self.common_step_counter * self.num_envs
        if self.domain_randomization_curriculum_steps == 0:
            curriculum_coeff = 1.0
        else:
            curriculum_coeff = min(global_step / self.domain_randomization_curriculum_steps, 1.0)


        roll_angle = ((torch.rand((nr_reset_envs,), device=self.sim.device) * 2) - 1.0) * math.pi * self.initial_state_roll_angle_factor * curriculum_coeff
        pitch_angle = ((torch.rand((nr_reset_envs,), device=self.sim.device) * 2) - 1.0) * math.pi * self.initial_state_pitch_angle_factor * curriculum_coeff
        yaw_angle = ((torch.rand((nr_reset_envs,), device=self.sim.device) * 2) - 1.0) * math.pi * self.initial_state_yaw_angle_factor * curriculum_coeff
        quaternion = axis_angle_to_quaternion(torch.stack([roll_angle, pitch_angle, yaw_angle], dim=1))
        joint_positions = self.robot.data.default_joint_pos[env_ids] * (1.0 + ((torch.rand((nr_reset_envs, self.nr_joints), device=self.sim.device) * 2) - 1.0) * self.initial_state_joint_nominal_position_factor * curriculum_coeff)

        # note that the randomization strength below line is dependent on the joint velocity range
        # some robots are very sensitive to the initial velocities, such as very tall humanoids
        # and the randomization could be overly strong for robots with super large joint_max_velocity
        # so it's good to clip the range to a constant, which is a hyper parameter
        joint_velocities = (self.joint_max_velocity[env_ids] * ((torch.rand((nr_reset_envs, self.nr_joints),
                                                                           device=self.sim.device) * 2) - 1.0)
                            * self.initial_state_joint_velocity_factor * curriculum_coeff)
        joint_velocities = joint_velocities.clip(-self.initial_state_joint_velocity_clip, self.initial_state_joint_velocity_clip)

        linear_velocity = ((torch.rand((nr_reset_envs, 3), device=self.sim.device) * 2) - 1.0) * self.initial_state_max_linear_velocity * curriculum_coeff
        angular_velocity = ((torch.rand((nr_reset_envs, 3), device=self.sim.device) * 2) - 1.0) * self.initial_state_max_angular_velocity * curriculum_coeff

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 3:7] = quaternion
        default_root_state[:, 7:10] = linear_velocity
        default_root_state[:, 10:] = angular_velocity

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        feet_indices, _ = self.robot.find_bodies(self.feet_contact_cfg.body_names, True)
        global_feet_pos = self.robot.data.body_pos_w[env_ids][:, feet_indices, :]
        lowest_feet = global_feet_pos[:, :, 2].min(dim=1).values
        root_state = self.robot.data.root_state_w[env_ids]
        root_state[:, 2] -= lowest_feet
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)

        self.previous_actions[env_ids] *= 0.0
        self.previous_feet_air_times[env_ids] *= 0.0

        self.action_history[env_ids] *= 0.0


    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions

        # Action delay
        self.action_history = torch.roll(self.action_history, -1, dims=1)
        self.action_history[:, -1] = self.actions
        current_nr_delay_steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.sim.device)
        current_nr_delay_steps[self.action_current_mixed] = torch.randint(0, self.max_nr_action_delay_steps + 1, (int(self.action_current_mixed.sum().item()),), dtype=torch.int32, device=self.sim.device)
        chosen_actions = self.action_history[torch.arange(self.num_envs, device=self.sim.device), -1 - current_nr_delay_steps]

        # PD control
        scaled_actions = chosen_actions * self.action_scaling_factor
        target_joint_positions = self.joint_nominal_positions + scaled_actions
        self.torques = self.p_gains * self.extrinsic_p_gain_factor * (target_joint_positions - self.robot.data.joint_pos + self.extrinsic_position_offset) \
                       - self.d_gains * self.extrinsic_d_gain_factor * self.robot.data.joint_vel
        self.torques = torch.clamp(self.torques * self.extrinsic_motor_strength, -self.joint_max_torque, self.joint_max_torque)


    def _apply_action(self):
        self.robot._joint_effort_target_sim = self.torques


    def handle_domain_randomization(self, all_envs=False):
        if not all_envs:
            env_randomization_mask = torch.rand((self.num_envs,), device=self.sim.device) < self.step_sampling_probability
        else:
            env_randomization_mask = torch.zeros((self.num_envs,), device=self.sim.device) < self.step_sampling_probability  # ugly, to avoid linting error
        nr_randomized_envs = env_randomization_mask.sum()

        global_step = self.common_step_counter * self.num_envs
        if self.domain_randomization_curriculum_steps == 0:
            curriculum_coeff = 1.0
        else:
            curriculum_coeff = min(global_step / self.domain_randomization_curriculum_steps, 1.0)


        # Action delay
        self.action_current_mixed[env_randomization_mask] = torch.rand((nr_randomized_envs,), device=self.sim.device) < self.mixed_action_delay_chance * curriculum_coeff
        
        # Control
        default_motor_stength = 1.0
        motor_strength_min = curriculum_coeff * self.motor_strength_min + (1.0 - curriculum_coeff) * default_motor_stength
        motor_strength_max = curriculum_coeff * self.motor_strength_max + (1.0 - curriculum_coeff) * default_motor_stength
        self.extrinsic_motor_strength[env_randomization_mask] = torch.rand((nr_randomized_envs, self.nr_joints), device=self.sim.device) * (motor_strength_max - motor_strength_min) + motor_strength_min
        
        default_p_gain_factor = 1.0
        p_gain_factor_min = curriculum_coeff * self.p_gain_factor_min + (1.0 - curriculum_coeff) * default_p_gain_factor
        p_gain_factor_max = curriculum_coeff * self.p_gain_factor_max + (1.0 - curriculum_coeff) * default_p_gain_factor
        self.extrinsic_p_gain_factor[env_randomization_mask] = torch.rand((nr_randomized_envs, self.nr_joints), device=self.sim.device) * (p_gain_factor_max - p_gain_factor_min) + p_gain_factor_min
        
        default_d_gain_factor = 1.0
        d_gain_factor_min = curriculum_coeff * self.d_gain_factor_min + (1.0 - curriculum_coeff) * default_d_gain_factor
        d_gain_factor_max = curriculum_coeff * self.d_gain_factor_max + (1.0 - curriculum_coeff) * default_d_gain_factor
        self.extrinsic_d_gain_factor[env_randomization_mask] = torch.rand((nr_randomized_envs, self.nr_joints), device=self.sim.device) * (d_gain_factor_max - d_gain_factor_min) + d_gain_factor_min
        
        p_law_position_offset_min = curriculum_coeff * self.p_law_position_offset_min
        p_law_position_offset_max = curriculum_coeff * self.p_law_position_offset_max
        self.extrinsic_position_offset[env_randomization_mask] = torch.rand((nr_randomized_envs, self.nr_joints), device=self.sim.device) * (p_law_position_offset_max - p_law_position_offset_min) + p_law_position_offset_min

        # Model
        env_randomization_indices = torch.nonzero(env_randomization_mask).flatten()
        middle_static_friction = (self.static_friction_min + self.static_friction_max) / 2.0
        static_friction_min = curriculum_coeff * self.static_friction_min + (1.0 - curriculum_coeff) * middle_static_friction
        static_friction_max = curriculum_coeff * self.static_friction_max + (1.0 - curriculum_coeff) * middle_static_friction
        middle_dynamic_friction = (self.dynamic_friction_min + self.dynamic_friction_max) / 2.0
        dynamic_friction_min = curriculum_coeff * self.dynamic_friction_min + (1.0 - curriculum_coeff) * middle_dynamic_friction
        dynamic_friction_max = curriculum_coeff * self.dynamic_friction_max + (1.0 - curriculum_coeff) * middle_dynamic_friction
        middle_restitution = (self.restitution_min + self.restitution_max) / 2.0
        restitution_min = curriculum_coeff * self.restitution_min + (1.0 - curriculum_coeff) * middle_restitution
        restitution_max = curriculum_coeff * self.restitution_max + (1.0 - curriculum_coeff) * middle_restitution
        self.randomize_rigid_body_material(self, env_randomization_indices, (static_friction_min, static_friction_max), (dynamic_friction_min, dynamic_friction_max), (restitution_min, restitution_max), 64, self.all_bodies_cfg)
        
        mdp.randomize_rigid_body_mass(self, env_randomization_indices, self.trunk_cfg, (self.added_trunk_mass_min * curriculum_coeff, self.added_trunk_mass_max * curriculum_coeff), "add")
        
        mdp.randomize_physics_scene_gravity(self, env_randomization_indices, (self.added_gravity_min * curriculum_coeff, self.added_gravity_max * curriculum_coeff), "add")
        
        middle_joint_friction = (self.joint_friction_min + self.joint_friction_max) / 2.0
        joint_friction_min = curriculum_coeff * self.joint_friction_min + (1.0 - curriculum_coeff) * middle_joint_friction
        joint_friction_max = curriculum_coeff * self.joint_friction_max + (1.0 - curriculum_coeff) * middle_joint_friction
        
        middle_joint_armature = (self.joint_armature_min + self.joint_armature_max) / 2.0
        joint_armature_min = curriculum_coeff * self.joint_armature_min + (1.0 - curriculum_coeff) * middle_joint_armature
        joint_armature_max = curriculum_coeff * self.joint_armature_max + (1.0 - curriculum_coeff) * middle_joint_armature
        mdp.randomize_joint_parameters(self, env_randomization_indices, self.all_joints_cfg, (joint_friction_min, joint_friction_max), (joint_armature_min, joint_armature_max), None, None, "abs")

        # Perturbations
        if not all_envs:
            env_perturbation_mask = torch.rand((self.num_envs,), device=self.sim.device) < self.step_sampling_probability
        else:
            env_perturbation_mask = torch.ones((self.num_envs,), device=self.sim.device) < self.step_sampling_probability  # ugly, to avoid linting error
        nr_perturbed_envs = env_perturbation_mask.sum()
        perturb_velocity_x_min, perturb_velocity_x_max = self.perturb_velocity_x_min * curriculum_coeff, self.perturb_velocity_x_max * curriculum_coeff
        perturb_velocity_y_min, perturb_velocity_y_max = self.perturb_velocity_y_min * curriculum_coeff, self.perturb_velocity_y_max * curriculum_coeff
        perturb_velocity_z_min, perturb_velocity_z_max = self.perturb_velocity_z_min * curriculum_coeff, self.perturb_velocity_z_max * curriculum_coeff
        perturb_velocity_x = torch.rand((nr_perturbed_envs,), device=self.sim.device) * (perturb_velocity_x_max - perturb_velocity_x_min) + perturb_velocity_x_min
        perturb_velocity_y = torch.rand((nr_perturbed_envs,), device=self.sim.device) * (perturb_velocity_y_max - perturb_velocity_y_min) + perturb_velocity_y_min
        perturb_velocity_z = torch.rand((nr_perturbed_envs,), device=self.sim.device) * (perturb_velocity_z_max - perturb_velocity_z_min) + perturb_velocity_z_min
        current_global_velocity = self.robot.data.root_state_w[env_perturbation_mask, 7:]
        current_global_velocity[:, 0] = torch.where(torch.rand((nr_perturbed_envs,), device=self.sim.device) < self.perturb_add_chance, perturb_velocity_x + current_global_velocity[:, 0] * self.perturb_additive_multiplier, perturb_velocity_x)
        current_global_velocity[:, 1] = torch.where(torch.rand((nr_perturbed_envs,), device=self.sim.device) < self.perturb_add_chance, perturb_velocity_y + current_global_velocity[:, 1] * self.perturb_additive_multiplier, perturb_velocity_y)
        current_global_velocity[:, 2] = torch.where(torch.rand((nr_perturbed_envs,), device=self.sim.device) < self.perturb_add_chance, perturb_velocity_z + current_global_velocity[:, 2] * self.perturb_additive_multiplier, perturb_velocity_z)
        env_perturbed_indices = torch.nonzero(env_perturbation_mask).flatten()
        self.robot.write_root_velocity_to_sim(current_global_velocity, env_perturbed_indices)


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.handle_domain_randomization()

        trunk_contact_sensor = self.scene.sensors[self.trunk_contact_cfg.name]
        trunk_contact = torch.any(torch.norm(trunk_contact_sensor.data.net_forces_w[:, self.trunk_contact_cfg.body_ids], dim=-1) > 1.0, dim=1)
        terminated = trunk_contact
        # terminated = torch.tensor([False] * self.num_envs, dtype=torch.bool, device=self.sim.device)  # for debugging

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated


    def _get_rewards(self) -> torch.Tensor:
        global_step = self.common_step_counter * self.num_envs
        if self.reward_curriculum_steps == 0:
            curriculum_coeff = 1.0
        else:
            curriculum_coeff = min(global_step / self.reward_curriculum_steps, 1.0)


        feet_contact_sensors = self.scene.sensors[self.feet_contact_cfg.name]
        feet_contacts = torch.norm(feet_contact_sensors.data.net_forces_w[:, self.feet_contact_cfg.body_ids], dim=-1) > 1.0

        feet_indices, _ = self.robot.find_bodies(self.feet_contact_cfg.body_names, True)
        global_feet_pos = self.robot.data.body_pos_w[:, feet_indices]
        local_feet_pos = math_utils.quat_rotate_inverse(self.robot.data.root_quat_w[:, None, :], global_feet_pos - self.robot.data.root_state_w[:, None, :3])

        reward, extras = compute_rewards(
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
            self.feet_y_distance_coeff,
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
            self.torques,
            self.goal_velocities,
            self.nominal_trunk_z,
            feet_contacts,
            self.previous_feet_air_times,
            self.feet_symmetry_pairs,
            local_feet_pos,
            self.feet_y_distance_target
        )

        self.extras = {"log": extras}  # sum(extras.values()) is reward

        self.previous_actions = self.actions.clone()
        self.previous_feet_air_times = feet_contact_sensors.data.current_air_time[:, self.feet_contact_cfg.body_ids].clone()

        return reward
    

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

    def update_goal_velocities(self):
        """
        Sample some robots and update the target velocities.
        This function should only be called by _get_observations function.
        """
        should_sample_new_goal_velocities = torch.rand((self.num_envs,), device=self.sim.device) < self.step_sampling_probability
        self.goal_velocities[should_sample_new_goal_velocities] = torch.rand((should_sample_new_goal_velocities.sum(), 3),
                                                                             device=self.sim.device).uniform_(-1.0, 1.0)
        return self.goal_velocities

    def _get_observations(self) -> dict:
        # Joint-specific observations
        joint_positions = self.robot.data.joint_pos - self.joint_nominal_positions
        joint_velocities = self.robot.data.joint_vel
        joint_previous_actions = self.actions

        # Feet-specific observations
        feet_contact_sensors = self.scene.sensors[self.feet_contact_cfg.name]
        feet_contacts = (torch.norm(feet_contact_sensors.data.net_forces_w[:, self.feet_contact_cfg.body_ids], dim=-1) > 1.0).float()
        feet_air_times = feet_contact_sensors.data.current_air_time[:, self.feet_contact_cfg.body_ids]

        # General observations
        trunk_linear_velocity = self.robot.data.root_lin_vel_b
        trunk_angular_velocity = self.robot.data.root_ang_vel_b

        goal_velocities = self.update_goal_velocities()

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
        
        # Add noise
        global_step = self.common_step_counter * self.num_envs
        if self.domain_randomization_curriculum_steps == 0:
            curriculum_coeff = 1.0
        else:
            curriculum_coeff = min(global_step / self.domain_randomization_curriculum_steps, 1.0)
        
        observation[:, self.joint_positions_obs_idx] += ((torch.rand_like(observation[:, self.joint_positions_obs_idx]) * 2) - 1) * self.joint_position_noise * curriculum_coeff
        observation[:, self.joint_velocities_obs_idx] += ((torch.rand_like(observation[:, self.joint_velocities_obs_idx]) * 2) - 1) * self.joint_velocity_noise * curriculum_coeff
        observation[:, self.trunk_angular_velocity_obs_idx] += ((torch.rand_like(observation[:, self.trunk_angular_velocity_obs_idx]) * 2) - 1) * self.trunk_angular_velocity_noise * curriculum_coeff
        observation[:, self.projected_gravity_vector_obs_idx] += ((torch.rand_like(observation[:, self.projected_gravity_vector_obs_idx]) * 2) - 1) * self.gravity_vector_noise * curriculum_coeff
        observation[:, self.feet_contact_obs_idx] = torch.where(torch.rand_like(observation[:, self.feet_contact_obs_idx]) < self.ground_contact_noise_chance * curriculum_coeff, 1.0 - observation[:, self.feet_contact_obs_idx], observation[:, self.feet_contact_obs_idx])
        observation[:, self.feet_air_time_obs_idx] = torch.where(torch.rand_like(observation[:, self.feet_air_time_obs_idx]) < self.contact_time_noise_chance * curriculum_coeff, observation[:, self.feet_air_time_obs_idx] + (((torch.rand_like(observation[:, self.feet_air_time_obs_idx]) * 2) - 1.0) * self.contact_time_noise_factor * curriculum_coeff * self.action_dt), observation[:, self.feet_air_time_obs_idx])

        # Dropout
        joint_dropout_mask = torch.rand((self.num_envs, self.nr_joints), device=self.sim.device) < self.joint_and_feet_dropout_chance * curriculum_coeff
        feet_dropout_mask = torch.rand((self.num_envs, self.nr_feet), device=self.sim.device) < self.joint_and_feet_dropout_chance * curriculum_coeff
        observation[:, self.joint_positions_obs_idx][joint_dropout_mask] = 0.0
        observation[:, self.joint_velocities_obs_idx][joint_dropout_mask] = 0.0
        observation[:, self.joint_previous_actions_obs_idx][joint_dropout_mask] = 0.0
        observation[:, self.feet_contact_obs_idx][feet_dropout_mask] = 0.0
        observation[:, self.feet_air_time_obs_idx][feet_dropout_mask] = 0.0

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

        if self.urma_initialized:
            urma_obs = self._get_urma_observations(observation)
            return {"policy": policy_observation, "critic": observation, "urma_obs": urma_obs}

        return {"policy": policy_observation, "critic": observation}

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
        # command = torch.cat([self.target_x_vel, self.target_y_vel, self.target_yaw_vel], dim=1)
        command = self.goal_velocities.clone()
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

    def _initialize_urma_observations(self):
        # Dynamic observations
        dynamic_joint_observations = torch.empty((self.num_envs, 0), device=self.sim.device)
        for i, joint_name in enumerate(self.joint_names):
            desc_vector = self.name_to_description_vector[joint_name]
            joint_pos_rel = torch.zeros((self.num_envs, 1), device=self.sim.device)
            joint_vel_rel = torch.zeros((self.num_envs, 1), device=self.sim.device)
            action = torch.zeros((self.num_envs, 1), device=self.sim.device)
            current_observation = torch.cat((desc_vector, joint_pos_rel, joint_vel_rel, action), dim=1)
            dynamic_joint_observations = torch.cat((dynamic_joint_observations, current_observation), dim=1)

        # General observations
        trunk_angular_velocity = torch.zeros((self.num_envs, 3), device=self.sim.device)
        goal_velocity = torch.zeros((self.num_envs, 3), device=self.sim.device)
        projected_gravity_vector = torch.zeros((self.num_envs, 3), device=self.sim.device)

        # General robot context
        gains_and_action_scaling_factor = ((self.gains_and_action_scaling_factor / torch.tensor(
            [100.0 / 2, 2.0 / 2, 0.8 / 2], device=self.sim.device)) - 1.0).repeat((self.num_envs, 1))
        mass = (self.mass / (170.0 / 2)) - 1.0
        robot_dimensions = ((self.robot_dimensions / (2.0 / 2)) - 1.0)

        observation = torch.cat([
            dynamic_joint_observations,
            trunk_angular_velocity,
            goal_velocity,
            projected_gravity_vector,
            gains_and_action_scaling_factor,
            mass,
            robot_dimensions,
        ], dim=1)

        return observation

    def _get_urma_observations(self, expert_observation: torch.Tensor) -> torch.Tensor:
        """
        Returns the observations for the URMA.
        """
        urma_obs = self.initial_observation.clone()

        # Update observations every step
        urma_obs[:, self.joint_positions_update_obs_idx] = expert_observation[:, self.joint_positions_obs_idx]
        urma_obs[:, self.joint_velocities_update_obs_idx] = expert_observation[:, self.joint_velocities_obs_idx]
        urma_obs[:, self.joint_previous_actions_update_obs_idx] = expert_observation[:, self.joint_previous_actions_obs_idx]
        urma_obs[:, self.trunk_angular_vel_update_obs_idx] = expert_observation[:, self.trunk_angular_velocity_obs_idx]
        urma_obs[:, self.goal_velocity_update_obs_idx] = expert_observation[:, self.goal_velocities_obs_idx]
        urma_obs[:, self.projected_gravity_update_obs_idx] = expert_observation[:, self.projected_gravity_vector_obs_idx]

        return urma_obs


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
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    # From: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L498

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
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
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


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
    feet_y_distance_coeff: float,
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
    nominal_trunk_z: float,
    feet_contacts: torch.Tensor,
    feet_air_times: torch.Tensor,
    feet_symmetry_pairs: torch.Tensor,
    local_feet_pos: torch.Tensor,
    feet_y_distance_target: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
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
    linear_velocity_reward = curriculum_coeff * z_velocity_coeff * -z_velocity_squared

    # Angular velocity reward
    angular_velocity_norm = torch.sum(torch.square(current_local_angular_velocity[:, :2]), dim=1)
    angular_velocity_reward = curriculum_coeff * pitch_roll_vel_coeff * -angular_velocity_norm

    # Angular position reward
    orientation_euler = quaternion_to_axis_angle(orientation_quat)
    pitch_roll_position_norm = torch.sum(torch.square(orientation_euler[:, :2]), dim=1)
    angular_position_reward = curriculum_coeff * pitch_roll_pos_coeff * -pitch_roll_position_norm

    # Joint nominal position difference reward
    actuator_joint_nominal_diff_norm = torch.mean(torch.square(joint_positions[:, actuator_joint_nominal_diff_joints] - joint_nominal_positions[:, actuator_joint_nominal_diff_joints]), dim=1)
    actuator_joint_nominal_diff_reward = curriculum_coeff * actuator_joint_nominal_diff_coeff * -actuator_joint_nominal_diff_norm

    # Joint position limit reward
    lower_limit_penalty = -torch.minimum(joint_positions - joint_position_soft_lower_limits, torch.tensor(0.0, device=joint_positions.device)).mean(dim=1)
    upper_limit_penalty = torch.maximum(joint_positions - joint_position_soft_upper_limits, torch.tensor(0.0, device=joint_positions.device)).mean(dim=1)
    joint_position_limit_reward = curriculum_coeff * joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)

    # Joint acceleration reward
    acceleration_norm = torch.mean(torch.square(joint_accelerations), dim=1)
    acceleration_reward = curriculum_coeff * joint_acceleration_coeff * -acceleration_norm

    # Joint torque reward
    torque_norm = torch.mean(torch.square(joint_torques), dim=1)
    torque_reward = curriculum_coeff * joint_torque_coeff * -torque_norm

    # Action rate reward
    action_rate_norm = torch.mean(torch.square(actions - previous_actions), dim=1)
    action_rate_reward = curriculum_coeff * action_rate_coeff * -action_rate_norm

    # Walking height reward
    trunk_z = linear_position[:, 2]
    height_difference_squared = (trunk_z - nominal_trunk_z) ** 2
    base_height_reward = curriculum_coeff * base_height_coeff * -height_difference_squared

    # Air time reward
    air_time_reward = curriculum_coeff * air_time_coeff * torch.sum(feet_contacts.float() * (feet_air_times - 0.5), dim=1)

    # Symmetry air reward
    symmetry_air_violations = torch.sum(~feet_contacts[:, feet_symmetry_pairs[:, 0]] & ~feet_contacts[:, feet_symmetry_pairs[:, 1]], dim=1)
    symmetry_air_reward = curriculum_coeff * symmetry_air_coeff * -symmetry_air_violations

    # Feet y distance reward
    feet_y_distance = torch.abs(local_feet_pos[:, feet_symmetry_pairs[:, 0], 1] - local_feet_pos[:, feet_symmetry_pairs[:, 1], 1]).mean(dim=1)
    feet_y_distance_from_target_norm = (feet_y_distance - feet_y_distance_target) ** 2
    feet_y_distance_reward = curriculum_coeff * feet_y_distance_coeff * -feet_y_distance_from_target_norm

    # Total reward
    tracking_reward = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward
    reward_penalty = linear_velocity_reward + angular_velocity_reward + angular_position_reward + actuator_joint_nominal_diff_reward + \
                     joint_position_limit_reward + acceleration_reward + torque_reward + action_rate_reward + \
                     base_height_reward + air_time_reward + symmetry_air_reward + feet_y_distance_reward
    reward = tracking_reward + reward_penalty
    reward = torch.maximum(reward, torch.tensor(0.0, device=reward.device))

    extras = {
        "reward/track_xy_vel_cmd": tracking_xy_velocity_command_reward.mean(),
        "reward/track_yaw_vel_cmd": tracking_yaw_velocity_command_reward.mean(),
        "reward/linear_velocity": linear_velocity_reward.mean(),
        "reward/angular_velocity": angular_velocity_reward.mean(),
        "reward/angular_position": angular_position_reward.mean(),
        "reward/joint_nominal_diff": actuator_joint_nominal_diff_reward.mean(),
        "reward/joint_position_limit": joint_position_limit_reward.mean(),
        "reward/joint_acceleration": acceleration_reward.mean(),
        "reward/joint_torque": torque_reward.mean(),
        "reward/action_rate": action_rate_reward.mean(),
        "reward/base_height": base_height_reward.mean(),
        "reward/air_time": air_time_reward.mean(),
        "reward/symmetry_air": symmetry_air_reward.mean(),
        "reward/feet_y_distance": feet_y_distance_reward.mean(),
    }

    return reward, extras
