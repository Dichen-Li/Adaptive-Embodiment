import os

def generate_robot_class_definitions(base_dir, output_file):
    """
    Reads robot folders in base_dir and generates class definitions for robot configs.
    """
    # Collect and sort robot folders
    robot_folders = sorted(
        [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    )

    # Open output file for writing the generated class definitions
    with open(output_file, 'w') as f_out:
        for idx, _ in enumerate(robot_folders, start=1):
            # Generate the robot configuration name and class name
            cfg_name = f"GEN_DOG_1K_{idx}_CFG"  # Format like GEN_DOG_1K_308_CFG
            class_name = f"GenDog1K{idx}Cfg"  # Format like GenDog1K302

            # Generate the class definition
            class_definition = f"""
@configclass
class {class_name}(GenDogEnvCfg):
    robot: ArticulationCfg = {cfg_name}
"""
            # Write to the output file
            f_out.write(class_definition)
            print(f"Generated class definition for: {class_name} (Robot Config: {cfg_name})")

        print(f"Generated {len(robot_folders)} class definitions.")

# Example usage
base_dir = "exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0/gen_dogs"  # Replace with the actual directory containing robot folders
output_file = "generated_robot_classes.py"  # Output file for the generated class definitions
generate_robot_class_definitions(base_dir, output_file)
