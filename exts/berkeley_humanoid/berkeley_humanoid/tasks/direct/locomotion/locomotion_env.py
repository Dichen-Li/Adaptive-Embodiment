from __future__ import annotations

import torch
import math

import omni.isaac.core.utils.torch as torch_utils
from berkeley_humanoid.tasks.direct.locomotion.command_function import RandomCommands
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg

from omni.isaac.lab.sensors import ContactSensor

from .joint_position_controller import JointPositionAction


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

        # configs needed for reward calculation
        self.reward_cfgs = self.cfg.reward_cfgs
        for cfg in self.reward_cfgs.values():
            cfg.resolve(self.scene)

        self.controller = JointPositionAction(self, self.cfg.action_scale,
                                              use_default_offset=self.cfg.controller_use_offset)

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

    def _get_observations(self) -> dict:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b
        self.target_x_vel, self.target_y_vel, self.target_yaw_vel = self.command_generator.get_next_command()
        joint_pos_rel = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel_rel = self.robot.data.joint_vel - self.robot.data.default_joint_vel
        actions = self.actions  # actions at prev step
        # base_lin_vel, base_ang_vel, joint_pos_rel, joint_vel_rel could explode

        obs = torch.cat(
            [base_lin_vel, base_ang_vel, projected_gravity_b, self.target_x_vel, self.target_y_vel,
             self.target_yaw_vel, joint_pos_rel, joint_vel_rel, actions], dim=1
        )

        observations = {"policy": obs}

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
