import os
import re
import json
import mujoco
import numpy as np
from xml.etree import ElementTree as ET

# Function to read joint connectivity from the URDF file
def read_urdf_joint_connectivity(urdf_path):
    """
    Reads a URDF file and computes the number of direct child joints for each joint.

    Args:
        urdf_path (str): Path to the URDF file.

    Returns:
        dict: A mapping of joint names to the number of direct child joints.
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Map from parent link to child joints
        joint_parent_map = {}
        for joint in root.findall(".//joint"):
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            joint_name = joint.get("name")
            if parent and joint_name:
                if parent not in joint_parent_map:
                    joint_parent_map[parent] = []
                joint_parent_map[parent].append(joint_name)

        # Count direct children for each joint
        joint_child_counts = {joint: len(children) for joint, children in joint_parent_map.items()}
        return joint_child_counts

    except ET.ParseError as e:
        raise ValueError(f"Error parsing URDF file: {e}")

# Function to compute nominal joint positions based on regex matching
def compute_nominal_joint_positions(joint_names, nominal_positions_cfg):
    """
    Computes nominal joint positions for each joint based on regex patterns from configuration.

    Args:
        joint_names (list): List of joint names.
        nominal_positions_cfg (dict): Configuration with regex patterns and their respective nominal positions.

    Returns:
        np.ndarray: Nominal joint positions in the correct order.
    """
    nominal_positions = np.zeros(len(joint_names))
    for idx, joint in enumerate(joint_names):
        for pattern, position in nominal_positions_cfg.items():
            if re.match(pattern, joint):
                nominal_positions[idx] = position
                break
    return nominal_positions

# Function to read and parse train configuration file
def read_train_config(train_cfg_path):
    """
    Reads and parses the train configuration file.

    Args:
        train_cfg_path (str): Path to the train_cfg.json file.

    Returns:
        dict: Parsed configuration data.
    """
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Train config file not found: {train_cfg_path}")

    with open(train_cfg_path, "r") as f:
        return json.load(f)

# Function to compute the description vector for joints
def compute_joint_description(data, joint_names, joint_nr_direct_child_joints, robot_dimensions, trunk_pos, mins):
    """
    Computes description vectors for all joints.

    Args:
        data (MjData): Mujoco simulation data object.
        joint_names (list): List of joint names.
        joint_nr_direct_child_joints (np.ndarray): Number of direct child joints.
        robot_dimensions (np.ndarray): Dimensions of the robot (length, width, height).
        trunk_pos (MjData.body): Position of the trunk in simulation.
        mins (np.ndarray): Minimum positions along x, y, z.

    Returns:
        dict: A mapping of joint names to their description vectors.
    """
    name_to_description_vector = {}

    for i, joint_name in enumerate(joint_names):
        joint = data.joint(joint_name)
        relative_joint_pos = joint.xanchor - trunk_pos.xpos
        relative_joint_pos = np.matmul(trunk_pos.xmat.reshape(3, 3).T, relative_joint_pos)
        relative_joint_pos_normalized = (relative_joint_pos - mins) / robot_dimensions
        joint_axis = joint.xaxis
        relative_joint_axis = np.matmul(trunk_pos.xmat.reshape(3, 3).T, joint_axis)

        name_to_description_vector[joint_name] = {
            "relative_joint_pos_normalized": (relative_joint_pos_normalized / 0.5) - 1.0,
            "relative_joint_axis": relative_joint_axis,
            "joint_nr_direct_child_joints": joint_nr_direct_child_joints[i] - 1.0,
            "robot_dimensions": (robot_dimensions / (2.0 / 2)) - 1.0
        }

    return name_to_description_vector

# Main function to process robot configuration
def process_robot_config(base_dir):
    """
    Process robot configuration to compute required values and generate description vectors.

    Args:
        base_dir (str): Path to the robot configuration directory.
    """
    urdf_path = os.path.join(base_dir, "robot.urdf")
    train_cfg_path = os.path.join(base_dir, "train_cfg.json")

    # Step 1: Read URDF and train configuration
    joint_child_counts = read_urdf_joint_connectivity(urdf_path)
    train_cfg = read_train_config(train_cfg_path)

    # Step 2: Extract values from train configuration
    foot_names = train_cfg.get("foot_names", [])
    joint_names = train_cfg.get("joint_names", [])
    nominal_joint_positions_cfg = train_cfg.get("nominal_joint_positions", {})

    # Step 3: Compute required values
    initial_drop_height = train_cfg.get("drop_height", 0.0)
    nominal_joint_positions = compute_nominal_joint_positions(joint_names, nominal_joint_positions_cfg)
    joint_nr_direct_child_joints = np.array([joint_child_counts.get(joint, 0) for joint in joint_names])

    # Step 4: Initialize Mujoco simulation and compute descriptions
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    trunk_pos = data.body("trunk")
    mins = np.min(data.geom_xpos, axis=0)  # Simplified for demonstration
    robot_dimensions = np.array([1.0, 1.0, 1.0])  # Placeholder dimensions

    name_to_description_vector = compute_joint_description(
        data, joint_names, joint_nr_direct_child_joints, robot_dimensions, trunk_pos, mins
    )

    return {
        "foot_names": foot_names,
        "joint_names": joint_names,
        "initial_drop_height": initial_drop_height,
        "nominal_joint_positions": nominal_joint_positions,
        "joint_nr_direct_child_joints": joint_nr_direct_child_joints,
        "name_to_description_vector": name_to_description_vector
    }

# Example usage
base_dir = "/path/to/robot/config"  # Replace with actual path
robot_config = process_robot_config(base_dir)
print(robot_config)
