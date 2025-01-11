import os
import json


def safe_get_with_warning(data, key, actuator_name, field_name, folder_name):
    """
    Safely fetches a value from a dictionary with a warning if the key is missing.
    Args:
        data (dict): The dictionary to fetch the value from.
        key (str): The key to look for in the dictionary.
        actuator_name (str): The name of the actuator (e.g., "legs", "feet").
        field_name (str): The name of the field being queried (e.g., "velocity_limit").
        folder_name (str): The name of the folder being processed.
    Returns:
        The value corresponding to the key if it exists, else None.
    Logs:
        A warning if the field is missing.
    """
    if key not in data:
        print(
            f"Warning: Missing '{field_name}' in actuator '{actuator_name}' for folder '{folder_name}'. Defaulting to None."
        )
        return None
    return data[key]

def format_field(field, indent=8):
    """
    Formats a field to handle both scalar values and dictionaries.
    Args:
        field: The field value, which can be a scalar (e.g., float, int) or a dictionary.
        indent (int): The indentation level for dictionary formatting.
    Returns:
        A formatted string representation of the field.
    """
    if isinstance(field, dict):
        # Properly format dictionaries with braces
        formatted_dict = ",\n".join(f'{" " * (indent + 4)}"{k}": {v:.2f}' for k, v in field.items())
        return f"{{\n{formatted_dict}\n{' ' * indent}}}"
    elif isinstance(field, (float, int)):
        # Return float as a scalar
        return f"{field:.2f}"
    return "None"  # Fallback for unexpected types

def process_actuator_config(actuator_name, actuator_data, folder_name):
    """
    Processes an actuator configuration and generates the corresponding code.
    Args:
        actuator_name (str): The name of the actuator (e.g., "legs", "feet").
        actuator_data (dict): The data for the actuator.
        folder_name (str): The name of the folder being processed.
    Returns:
        A string containing the generated code for the actuator.
    """
    joint_names_expr = safe_get_with_warning(
        actuator_data, "joint_names_expr", actuator_name, "joint_names_expr", folder_name
    )
    effort_limit = safe_get_with_warning(
        actuator_data, "effort_limit", actuator_name, "effort_limit", folder_name
    )
    velocity_limit = safe_get_with_warning(
        actuator_data, "velocity_limit", actuator_name, "velocity_limit", folder_name
    )
    stiffness = safe_get_with_warning(
        actuator_data, "stiffness", actuator_name, "stiffness", folder_name
    )
    damping = safe_get_with_warning(
        actuator_data, "damping", actuator_name, "damping", folder_name
    )
    armature = safe_get_with_warning(
        actuator_data, "armature", actuator_name, "armature", folder_name
    )

    # get string
    assert effort_limit is not None
    velocity_limit_str = f"{velocity_limit:.2f}" if velocity_limit is not None else None

    # Format fields appropriately
    stiffness_str = format_field(stiffness)
    damping_str = format_field(damping)
    armature_str = format_field(armature)

    return f"""
    "{actuator_name}": ImplicitActuatorCfg(
        joint_names_expr={joint_names_expr},
        effort_limit={effort_limit:.2f},
        velocity_limit={velocity_limit_str},
        stiffness={stiffness_str},
        damping={damping_str},
        armature={armature_str},
    )"""


def generate_code(base_dir, output_file):
    """
    Reads robot folders in base_dir, parses train_cfg.json, and generates configuration code.
    """
    # Collect and sort robot folders
    robot_folders = sorted(
        [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    )
    robot_folder_name = os.path.basename(base_dir)

    # Open output file for writing the generated code
    with open(output_file, 'w') as f_out:
        for robot_folder in robot_folders:
            # Define paths to train_cfg.json and USD file
            train_cfg_path = os.path.join(robot_folder, 'train_cfg_v2.json')
            usd_file_path = os.path.join(robot_folder, 'usd_file', 'robot.usd')

            # Skip if required files do not exist
            if not os.path.exists(train_cfg_path) or not os.path.exists(usd_file_path):
                print(f"Skipping {robot_folder}: Missing train_cfg.json or robot.usd")
                continue

            # Read train_cfg.json
            with open(train_cfg_path, 'r') as f:
                train_cfg = json.load(f)

            # Extract robot_name
            robot_name = train_cfg.get("robot_name")
            if not robot_name:
                raise KeyError(f"'robot_name' is missing in train_cfg.json for folder: {robot_folder}")

            # Generate CFG name
            cfg_name = f"{robot_name.upper()}_CFG"  # Consistent CFG naming

            # Extract actuator configurations
            actuators_config = train_cfg.get("actuators", {})
            actuators_code = [
                process_actuator_config(actuator_name, actuator_data, robot_folder)
                for actuator_name, actuator_data in actuators_config.items()
            ]
            actuators_code_str = ",\n".join(actuators_code)

            # Use the folder name to construct USD path
            folder_name = os.path.basename(robot_folder)
            usd_path = f'{{ISAAC_ASSET_DIR}}/Robots/GenBot1K-v0/{robot_folder_name}/{folder_name}/usd_file/robot.usd'

            # Generate code block
            code_block = f"""
actuators = {{
{actuators_code_str}
}}

{cfg_name} = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{usd_path}",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, {train_cfg.get("drop_height", 0.5):.2f}),
        joint_pos={{
            {", ".join(f'"{k}": {v:.2f}' for k, v in train_cfg.get("nominal_joint_positions", {}).items())}
        }},
        joint_vel={{".*": 0.0}},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)
"""
            # Write to output file
            f_out.write(code_block)
            print(f"Generated configuration for: {robot_name}")

    print(f"Successfully processed {len(robot_folders)} robot folders.")


# Example usage
base_dir = "/home/albert/Data/gen_embodiments_0110_no_usd/gen_humanoids"  # Replace with the actual directory containing robot folders
output_file = "articulation_cfgs.py"  # Output file for the generated code
generate_code(base_dir, output_file)
