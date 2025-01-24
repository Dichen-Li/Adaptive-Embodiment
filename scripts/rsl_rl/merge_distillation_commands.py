import os
import re

def merge_args(source_dirs, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    # Get all sh files grouped by their names across directories
    file_groups = {}
    for source_dir in source_dirs:
        for file_name in os.listdir(source_dir):
            if file_name.endswith(".sh"):
                if file_name not in file_groups:
                    file_groups[file_name] = []
                file_groups[file_name].append(os.path.join(source_dir, file_name))
    
    for file_name, paths in file_groups.items():
        train_set = []
        test_set = []
        other_args = f"--model urma --exp_name {file_name[:-3] + '_3robots_fake'} --batch_size 256 --lr 3e-4 --num_workers 0 --max_files_in_memory 5 --num_epochs 10 --gradient_acc_steps 1"
        
        # Read each file and extract train_set, test_set, and other arguments
        for path in paths:
            with open(path, "r") as file:
                content = file.read()
                
                # Use the heuristic to extract train_set and test_set
                train_match = re.search(r"--train_set\s+((?:(?! --).)+)", content)
                if train_match:
                    train_set.extend(train_match.group(1).strip().split())
                
                test_match = re.search(r"--test_set\s+((?:(?! --).)+)", content)
                if test_match:
                    test_set.extend(test_match.group(1).strip().split())
                
                # # Keep other args from one of the files
                # if other_args is None:
                #     other_args = re.sub(r"--train_set\s+[^\n]+", "", content)
                #     other_args = re.sub(r"--test_set\s+[^\n]+", "", other_args)
        
        # Combine the train_set and test_set args
        merged_content = f"python scripts/rsl_rl/run_distillation.py --train_set {' '.join(train_set)} --test_set {' '.join(test_set)} " + other_args
        
        # Save the new file in the target directory
        target_path = os.path.join(target_dir, file_name)
        with open(target_path, "w") as target_file:
            target_file.write(merged_content)

# Example usage
source_dirs = [
    "/home/liudai/hdd_0/projects/cross_em/bai/embodiment-scaling-law/scripts/rsl_rl/quadruped_jobs",
    "/home/liudai/hdd_0/projects/cross_em/bai/embodiment-scaling-law/scripts/rsl_rl/quadruped_jobs",
    # "/home/liudai/hdd_0/projects/cross_em/bai/embodiment-scaling-law/scripts/rsl_rl/humanoid_jobs",
    "/home/liudai/hdd_0/projects/cross_em/bai/embodiment-scaling-law/scripts/rsl_rl/hexapod_jobs",
]
target_dir = "/home/liudai/hdd_0/projects/cross_em/bai/embodiment-scaling-law/scripts/rsl_rl/3robots_jobs_fake"

os.makedirs(target_dir, exist_ok=True)
merge_args(source_dirs, target_dir)
