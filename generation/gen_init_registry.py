import os


def generate_robot_registration_with_custom_names(base_dir, output_file):
    """
    Reads robot folders in base_dir and generates robot config registration code with simplified names.
    """
    # Collect and sort robot folders
    robot_folders = sorted(
        [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    )

    # Open output file for writing the generated registration code
    with open(output_file, 'w') as f_out:
        # Write the comment header
        f_out.write('"""\nRegister customized robot configs.\n'
                    'Due to the large number of configs, ideally we want to automate the process\n"""\n\n')

        # Initialize the id_entry_pair dictionary
        f_out.write("id_entry_pair = {\n")

        for idx, _ in enumerate(robot_folders, start=1):
            # Generate the robot ID in the format GEN_DOG_1K_{i}
            robot_id = f"GenDog1K{idx}"

            # Assign a simple name for the configuration class
            cfg_name = f"GenDog1K{idx}Cfg"

            # Add entry to the dictionary
            f_out.write(f'    "{robot_id}": {cfg_name},\n')

        # Close the id_entry_pair dictionary
        f_out.write("}\n\n")

        # Write the registration loop
        f_out.write("for id, env_cfg_entry_point in id_entry_pair.items():\n")
        f_out.write("    rsl_rl_cfg_entry_point = f\"{agents.__name__}.gen_quadruped_1k_ppo_cfg:{id}PPORunnerCfg\"\n")
        f_out.write("    gym.register(\n")
        f_out.write("        id=id,\n")
        f_out.write("        entry_point=\"berkeley_humanoid.tasks.direct.humanoid:GenDirectEnv\",\n")
        f_out.write("        disable_env_checker=True,\n")
        f_out.write("        kwargs={\n")
        f_out.write("            \"env_cfg_entry_point\": env_cfg_entry_point,\n")
        f_out.write("            \"rsl_rl_cfg_entry_point\": rsl_rl_cfg_entry_point\n")
        f_out.write("        },\n")
        f_out.write("    )\n")

        print(f"Generated robot registration code for {len(robot_folders)} robots.")


# Example usage
base_dir = "exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0/gen_dogs"  # Replace with the actual directory containing robot folders
output_file = "generated_robot_registration.py"  # Output file for the generated registration code
generate_robot_registration_with_custom_names(base_dir, output_file)
