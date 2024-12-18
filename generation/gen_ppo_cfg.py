import os
import json

def generate_runner_configs_with_adapted_names(base_dir, output_file):
    """
    Reads robot folders and train_cfg.json files in base_dir, generates Runner Config classes with consistent naming.
    """
    # Collect and sort robot folders
    robot_folders = sorted(
        [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    )
    total_folders = len(robot_folders)

    # Open output file for writing the generated class definitions
    with open(output_file, 'w') as f_out:
        for idx, robot_folder in enumerate(robot_folders, start=1):
            # Define paths to train_cfg.json
            train_cfg_path = os.path.join(base_dir, robot_folder, "train_cfg.json")

            # Skip folders without train_cfg.json
            if not os.path.exists(train_cfg_path):
                print(f"Skipping {robot_folder}: Missing train_cfg.json")
                continue

            # Read train_cfg.json
            with open(train_cfg_path, 'r') as f:
                train_cfg = json.load(f)

            # Extract robot name from train_cfg.json
            robot_name = train_cfg.get("robot_name", None)
            if not robot_name:
                print(f"Skipping {robot_folder}: 'robot_name' not found in train_cfg.json.")
                continue

            # Process CFG name and class name
            cfg_name = f"{robot_name.replace('_', '').capitalize()}Cfg"  # Example: GenDog1K5Cfg
            class_name = f"{robot_name.replace('_', '').capitalize()}PPORunnerCfg"  # Example: GenDog1K5PPORunnerCfg

            # Use the original folder name (unchanged) as the experiment name
            experiment_name = robot_name.replace('_', '').capitalize() + '_' + robot_folder

            # Generate class definition
            class_definition = f"""
@configclass
class {class_name}(DefaultPPORunnerCfg):
    experiment_name = "{experiment_name}"
"""
            # Write to output file
            f_out.write(class_definition)
            # Print verbose progress
            print(f"Processing {idx}/{total_folders}: Generated runner config class for {class_name} (CFG: {cfg_name}, Experiment: {experiment_name})")

    print(f"Generated {total_folders} runner config classes.")


# Example usage
base_dir = "../exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0/gen_hexapods"  # Replace with the actual directory containing robot folders
output_file = "ppo_cfg.py"  # Output file for the generated class definitions
generate_runner_configs_with_adapted_names(base_dir, output_file)
