import os
import re
from argparse import ArgumentParser
from datetime import datetime


def extract_index(folder_name, keyword):
    """Extract the numeric index from a folder name using the given keyword."""
    pattern = rf"{keyword}(\d+)"
    match = re.match(pattern, folder_name)
    return int(match.group(1)) if match else None


def find_latest_subfolder(folder_path):
    """Find the most recent subfolder in the given folder based on timestamp."""
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not subfolders:
        return None
    return max(
        subfolders,
        key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"),
        default=None
    )


def analyze_logs(root_dir, keyword, max_index, min_epoch):
    """Analyze log directories and categorize them into missing, incomplete, and complete runs."""
    index_range = range(0, max_index)

    # Get all folders matching the pattern
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    folder_indices = {
        extract_index(f, keyword): f for f in folders if extract_index(f, keyword) is not None
    }

    # Initialize results
    missing_logs = [i for i in index_range if i not in folder_indices]
    incomplete_logs = []
    complete_logs = []

    for idx in sorted(folder_indices.keys()):
        folder = folder_indices[idx]
        folder_path = os.path.join(root_dir, folder)

        # Find the most recent subfolder
        latest_folder = find_latest_subfolder(folder_path)
        if not latest_folder:
            incomplete_logs.append((idx, folder))
            continue

        latest_folder_path = os.path.join(folder_path, latest_folder)
        pt_files = [f for f in os.listdir(latest_folder_path) if f.endswith(".pt")]

        # Validate .pt files based on min_epoch
        valid_files = [f for f in pt_files if int(f.split("_")[1].split(".")[0]) >= min_epoch]
        if valid_files:
            complete_logs.append((idx, folder))
        else:
            incomplete_logs.append((idx, folder))

    # Sort results by index
    missing_logs.sort()
    incomplete_logs.sort(key=lambda x: x[0])
    complete_logs.sort(key=lambda x: x[0])

    # Return results
    return {
        "complete_logs": complete_logs,
        "incomplete_logs": incomplete_logs,
        "missing_logs": missing_logs,
    }


def print_results(results):
    """Print categorized log results."""
    print("=== Missing Directories ===")
    if results["missing_logs"]:
        print(f"Logs missing for indices: {results['missing_logs']}")
    else:
        print("No missing logs.")

    print("\n=== Incomplete Runs ===")
    if results["incomplete_logs"]:
        print(f"Incomplete logs (no .pt files after specified min_epoch):")
        for idx, log in results["incomplete_logs"]:
            print(f"- Index: {idx}, Directory: {log}")
    else:
        print("No incomplete logs.")

    print("\n=== Complete Runs ===")
    if results["complete_logs"]:
        print(f"Complete logs (contain valid .pt files):")
        for idx, log in results["complete_logs"]:
            print(f"- Index: {idx}, Directory: {log}")
    else:
        print("No complete logs.")


if __name__ == "__main__":
    # Argument parsing
    parser = ArgumentParser(description="Analyze log directories for completeness based on checkpoints.")
    parser.add_argument("--root", type=str, default="/bai-fast-vol/code/embodiment-scaling-law/logs/rsl_rl",
                        help="Root directory containing the log files. Default: /bai-fast-vol/code/embodiment-scaling-law/logs/rsl_rl")
    parser.add_argument("--keyword", type=str, default="Gendog",
                        help="Keyword to match log directories (e.g., 'Gendog'). Default: Gendog")
    parser.add_argument("--max-index", type=int, default=308,
                        help="Maximum index (exclusive) for log directories. Default: 308")
    parser.add_argument("--min-epoch", type=int, default=3000,
                        help="Minimum epoch to consider a run complete. Default: 3000")
    args = parser.parse_args()

    # Analyze logs and print results
    results = analyze_logs(args.root, args.keyword, args.max_index, args.min_epoch)
    print_results(results)
