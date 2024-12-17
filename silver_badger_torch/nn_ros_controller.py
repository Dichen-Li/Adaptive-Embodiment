import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from policy import get_policy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from hb40_commons.msg import BridgeData
from hb40_commons.msg import JointCommand
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class RobotHandler(Node):
    def __init__(self):
        super().__init__("robot_handler")

        self.is_real_robot = False
        self.is_tuda_robot = True

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.bridge_data_subscription = self.create_subscription(
            BridgeData,
            "/hb40/bridge_data",
            self.bridge_data_callback,
            qos_profile=qos_profile)
        
        self.velocity_command_subscription = self.create_subscription(
            Twist,
            "/hb40/velocity_command",
            self.velocity_command_callback,
            qos_profile=qos_profile)
        
        self.joystick_subscription = self.create_subscription(
            Joy,
            "/hb40/joy",
            self.joystick_callback,
            qos_profile=qos_profile)
        
        self.joint_commands_publisher = self.create_publisher(
            JointCommand,
            "/hb40/joint_command" if self.is_real_robot else "/hb40/joint_commandHighPrio",
            qos_profile=qos_profile)
        timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.joint_positions = np.zeros(13)
        self.joint_velocities = np.zeros(13)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.angular_velocity = np.zeros(3)
        self.x_goal_velocity = 0.0
        self.y_goal_velocity = 0.0
        self.yaw_goal_velocity = 0.0

        self.kp = [20.0,] * 13
        self.kd = [0.5,] * 13
        self.scaling_factor = 0.25
        self.nominal_joint_positions = np.array([
            -0.1, 0.8, -1.5,
            0.1, -0.8, 1.5,
            -0.1, -1.0, 1.5,
            0.1, 1.0, -1.5,
            0.0
        ])

        self.mask_from_real_to_sim = [12, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
        self.mask_from_sim_to_real = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 0]

        self.previous_action = np.zeros(13)

        dynamic_joint_description_path = "bg_dynamic_joint_description.npy"
        dynamic_foot_description_path = "bg_dynamic_foot_description.npy"
        general_policy_state_second_part_path = "bg_general_policy_state.npy"
        self.dynamic_joint_description = torch.tensor(np.load(dynamic_joint_description_path), dtype=torch.float32)
        self.dynamic_foot_description = torch.tensor(np.load(dynamic_foot_description_path), dtype=torch.float32)
        self.general_policy_state_second_part = np.load(general_policy_state_second_part_path)

        self.policy = get_policy()

        self.nn_active = False

        print(f"Robot ready. Using device: CPU")


    def bridge_data_callback(self, msg):
        self.joint_positions = np.array(msg.joint_position)
        self.joint_velocities = np.array(msg.joint_velocity)
        if self.is_real_robot and self.is_tuda_robot:
            self.joint_positions[-1] *= -1.0
            self.joint_velocities[-1] *= -1.0
        self.orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])


    def velocity_command_callback(self, msg):
        self.x_goal_velocity = msg.linear.x
        self.y_goal_velocity = msg.linear.y
        self.yaw_goal_velocity = msg.angular.z * 0.5  # Topic gives -2 to 2, but we want -1 to 1


    def joystick_callback(self, msg):
        if msg.buttons[0] == 1 and self.nn_active:
            self.nn_active = False
        elif msg.buttons[1] == 1 and not self.nn_active:
            self.nn_active = True


    def timer_callback(self):
        if not self.nn_active:
            return
        
        transposed_trunk_rotation_matrix = R.from_quat(self.orientation).as_matrix().T
        qpos = self.joint_positions - self.nominal_joint_positions
        qvel = self.joint_velocities
        ang_vel = self.angular_velocity
        projected_gravity_vector = np.matmul(transposed_trunk_rotation_matrix, np.array([0.0, 0.0, -1.0]))

        dynamic_joint_state = np.concatenate([
            qpos[self.mask_from_real_to_sim].reshape(1, 13, 1) / 4.6,
            qvel[self.mask_from_real_to_sim].reshape(1, 13, 1) / 35.0,
            self.previous_action[self.mask_from_real_to_sim].reshape(1, 13, 1) / 10.0
        ], axis=2)

        dynamic_foot_state = np.zeros((1, 4, 2))
        dynamic_foot_state[:, :, 0] = (dynamic_foot_state[:, :, 0] / 0.5) - 1.0
        dynamic_foot_state[:, :, 1] = np.clip((dynamic_foot_state[:, :, 1] / (5.0 / 2)) - 1.0, -1.0, 1.0)

        general_policy_state = np.concatenate([
            [np.clip(ang_vel / 50.0, -1.0, 1.0)],
            [[self.x_goal_velocity, self.y_goal_velocity, self.yaw_goal_velocity]],
            [projected_gravity_vector],
            self.general_policy_state_second_part
        ], axis=1)

        with torch.no_grad():
            dynamic_joint_state = torch.tensor(dynamic_joint_state, dtype=torch.float32)
            dynamic_foot_state = torch.tensor(dynamic_foot_state, dtype=torch.float32)
            general_policy_state = torch.tensor(general_policy_state, dtype=torch.float32)
            action = self.policy(self.dynamic_joint_description, dynamic_joint_state, self.dynamic_foot_description, dynamic_foot_state, general_policy_state)
        action = action.numpy()[0, self.mask_from_sim_to_real]

        target_joint_positions = self.nominal_joint_positions + self.scaling_factor * action
        if self.is_real_robot and self.is_tuda_robot:
            target_joint_positions[-1] *= -1.0

        torques = self.kp * (target_joint_positions - self.joint_positions) - self.kd * self.joint_velocities

        joint_command_msg = JointCommand()
        joint_command_msg.header.stamp = self.get_clock().now().to_msg()
        joint_command_msg.source_node = "nn_controller"
        joint_command_msg.kp = [0.0,] * 13
        joint_command_msg.kd = [0.0,] * 13
        joint_command_msg.t_pos = self.joint_positions.tolist()
        joint_command_msg.t_vel = [0.0,] * 13
        joint_command_msg.t_trq = torques.tolist()

        self.joint_commands_publisher.publish(joint_command_msg)

        self.previous_action = action


def main(args=None):
    rclpy.init(args=args)
    robot_handler = RobotHandler()
    rclpy.spin(robot_handler)
    robot_handler.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
