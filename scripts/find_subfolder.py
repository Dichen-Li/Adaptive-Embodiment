import os
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

def find_latest_folder_with_expected_subfolder(base_path, expected_folder_name):
    """
    Finds folders where the latest subfolder contains the expected folder.
    """
    results = {}

    # Iterate through all folders in the base path
    for root, dirs, files in os.walk(base_path):
        # Extract datetime-parsable subfolders
        datetime_folders = {
            folder: parse_datetime(folder)
            for folder in dirs
            if parse_datetime(folder) is not None
        }

        # Skip if no datetime-parsable folders
        if not datetime_folders:
            continue

        # Find the latest folder by datetime
        latest_folder = max(datetime_folders, key=lambda k: datetime_folders[k])
        latest_folder_path = os.path.join(root, latest_folder)

        # Check if the expected folder exists in the latest folder
        if os.path.isdir(os.path.join(latest_folder_path, expected_folder_name)):
            # Add the parent folder to results
            parent_folder_name = os.path.basename(root)
            results[parent_folder_name] = latest_folder_path

    # Sort the results based on the numeric part of the folder name
    sorted_keys = sorted(results.keys(), key=lambda k: int(k.split("_")[0].replace("Gendog", "")))

    # Print sorted results
    for key in sorted_keys:
        print(key)
    
    # some statistics
    print(f'Summary: {len(sorted_keys)} folders are found.')

if __name__ == "__main__":
    # Specify the base path and the expected folder name
    base_path = "/bai-fast-vol/code/embodiment-scaling-law/logs/rsl_rl"
    expected_folder_name = "h5py_record"

    # Run the function
    find_latest_folder_with_expected_subfolder(base_path, expected_folder_name)
