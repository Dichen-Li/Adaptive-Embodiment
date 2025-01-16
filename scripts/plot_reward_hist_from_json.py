import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime


def parse_datetime(folder_name):
    """
    Parses a folder name in the format 'YYYY-MM-DD_HH-MM-SS' and returns a datetime object.
    Returns None if the parsing fails.
    """
    try:
        return datetime.strptime(folder_name, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training results from TensorBoard logs.")
    parser.add_argument("--log_path", type=str, default="./logs/rsl_rl", help="Path to the RSL RL logs.")
    parser.add_argument("--save_path", type=str, default="reward_his.jpg", help="Path to save the histogram.")
    parser.add_argument("--keyword", type=str, default="Genhumanoid")

    args = parser.parse_args()

    # enumerate all folders in the log_path
    finished_returns = []
    for folder in os.listdir(args.log_path):
        subfolders = os.listdir(os.path.join(args.log_path, folder))
        # Find the latest date-time subfolder
        dated_subfolders = [(subfolder, parse_datetime(subfolder[:19])) for subfolder in subfolders]
        if len(dated_subfolders) == 0:
            continue
        dated_subfolders = [(name, dt) for name, dt in dated_subfolders if dt is not None]
        # Sort subfolders by datetime and take the latest
        latest_subfolder = max(dated_subfolders, key=lambda x: x[1])[0]
        
        json_path = os.path.join(args.log_path, folder, latest_subfolder, "h5py_record/reward_log_file.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            finished_returns.append(v["average_return"])

    plt.hist(finished_returns, bins=10, edgecolor='black')
    plt.title(f"Histogram of Return")
    plt.xlabel("Mean Return")
    plt.ylabel("Frequency")

    if args.save_path:
        plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {args.save_path}")