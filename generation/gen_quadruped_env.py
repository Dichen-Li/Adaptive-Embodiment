import os
import json


def generate_robot_classes_from_config(base_dir, output_file):
    """
    Reads robot folders and train_cfg.json files in base_dir and generates class definitions based on robot names in the config file.
    """
    # Collect and sort robot folders
    robot_folders = sorted(
        [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    )

    # Predefined reward configuration structure
    reward_cfg_structure = {
        'feet_ground_contact_cfg': ('contact_sensor', 'body_names'),
        'feet_ground_asset_cfg': ('robot', 'body_names'),
        'undesired_contact_cfg': ('contact_sensor', 'body_names'),
        'joint_hip_cfg': ('robot', 'joint_names'),
        'joint_knee_cfg': ('robot', 'joint_names'),
        'illegal_contact_cfg': ('contact_sensor', 'body_names'),
    }

    # Open output file for writing the generated class definitions
    with open(output_file, 'w') as f_out:
        for idx, robot_folder in enumerate(robot_folders, start=1):
            # Paths for the train_cfg.json and robot configuration
            train_cfg_path = os.path.join(base_dir, robot_folder, "train_cfg.json")
            if not os.path.exists(train_cfg_path):
                print(f"Skipping {robot_folder}: train_cfg.json not found.")
                continue

            # Load the train_cfg.json
            with open(train_cfg_path, "r") as f:
                train_cfg = json.load(f)

            # Extract robot name from the configuration
            robot_name = train_cfg.get("robot_name", None)
            if not robot_name:
                print(f"Skipping {robot_folder}: 'robot_name' not found in train_cfg.json.")
                continue

            # Capitalize and format the robot name for the class name
            class_name = robot_name.replace("_", "").capitalize() + "Cfg"
            cfg_name = f"{robot_name.upper()}_CFG"

            # Extract action space
            action_space = train_cfg.get("action_space", 0)

            # Extract and format reward configurations
            reward_cfgs = {}
            for key, (first_arg, second_arg) in reward_cfg_structure.items():
                # Retrieve the reward configuration values from the JSON
                config_values = train_cfg.get("reward_cfgs", {}).get(key, None)
                if config_values is None or (isinstance(config_values, list) and not config_values):
                    # Handle empty or missing values
                    names_str = "[]"
                else:
                    # Extract body_names or joint_names
                    names = config_values if isinstance(config_values, list) else [config_values]
                    names_str = "[" + ", ".join(f"'{v}'" for v in names) + "]"

                # Add the reward configuration
                reward_cfgs[key] = f"SceneEntityCfg('{first_arg}', {second_arg}={names_str})"

            # Format reward_cfgs as a string
            formatted_reward_cfgs_str = ",\n".join(
                f"        '{key}': {value}" for key, value in reward_cfgs.items()
            )

            # Generate the class definition
            class_definition = (
                f"@configclass\n"
                f"class {class_name}(GenDogEnvCfg):\n"
                f"    action_space = {action_space}\n"
                f"    robot: ArticulationCfg = {cfg_name}\n"
                f"    reward_cfgs = {{\n"
                f"{formatted_reward_cfgs_str}\n"
                f"    }}\n\n"
            )

            # Write to the output file
            f_out.write(class_definition)
            print(f"Generated class definition for: {class_name} (Robot Config: {cfg_name})")

        print(f"Generated {len(robot_folders)} class definitions.")


# Example usage
base_dir = "../exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0/gen_dogs"  # Replace with the actual directory containing robot folders
output_file = "robot_classes.py"  # Output file for the generated class definitions
generate_robot_classes_from_config(base_dir, output_file)
