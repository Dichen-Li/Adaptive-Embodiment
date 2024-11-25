import os

def generate_runner_configs_with_verbose(base_dir, output_file):
    """
    Reads robot folders in base_dir, generates Runner Config classes with consistent naming.
    """
    # Collect and sort robot folders
    robot_folders = sorted(
        [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    )
    total_folders = len(robot_folders)

    # Open output file for writing the generated class definitions
    with open(output_file, 'w') as f_out:
        for idx, robot_folder in enumerate(robot_folders, start=1):
            # Generate class name similar to GEN_DOG_1K_{idx}_CFG
            cfg_name = f"GenDog1K{idx}"

            # Use the original folder name (with underscores) for the experiment name
            experiment_name = robot_folder

            # Generate class name
            class_name = f"{cfg_name}PPORunnerCfg"

            # Generate class definition
            class_definition = f"""
@configclass
class {class_name}(HumanoidPPORunnerCfg):
    experiment_name = "{experiment_name}"
"""
            # Write to output file
            f_out.write(class_definition)
            # Print verbose progress
            print(f"Processing {idx}/{total_folders}: Generated runner config class for {cfg_name} (Experiment: {experiment_name})")


# Example usage
base_dir = "exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0/gen_dogs"  # Replace with the actual directory containing robot folders
output_file = "gen_quadrupeds_ppo.py"  # Output file for the generated class definitions
generate_runner_configs_with_verbose(base_dir, output_file)
