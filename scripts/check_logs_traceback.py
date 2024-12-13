import os
import re
from argparse import ArgumentParser

def extract_index(file_name, keyword):
    """Extract the numeric index from the file name using the given keyword."""
    pattern = rf"{keyword}(\d+).*"
    match = re.match(pattern, file_name)
    return int(match.group(1)) if match else None

def check_file_for_errors(file_path, traceback_pattern, error_pattern, num_lines):
    """Check if the file contains errors and capture the last `num_lines` of the file."""
    has_traceback = False
    last_lines = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if traceback_pattern in line:
                    has_traceback = True
                if has_traceback and error_pattern in line:
                    last_lines = lines[-num_lines:]  # Capture the last `num_lines`
                    return True, last_lines
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return False, last_lines

def analyze_logs(root_dir, keyword, max_index, num_lines, traceback_pattern="Traceback", error_pattern="Error"):
    """Analyze log files and categorize them into errors, missing, and good files."""
    log_files = [f for f in os.listdir(root_dir) if re.match(rf"{keyword}\d+.*", f)]
    file_indices = {
        extract_index(f, keyword): f
        for f in log_files
        if extract_index(f, keyword) is not None
    }

    # Determine the range of indices to analyze
    index_range = range(0, max_index)

    # Categorize files
    missing_files = []
    good_files = []
    error_logs = []

    for idx in index_range:
        if idx not in file_indices:
            missing_files.append(idx)
            continue

        log_file = file_indices[idx]
        log_file_path = os.path.join(root_dir, log_file)

        has_error, last_lines = check_file_for_errors(log_file_path, traceback_pattern, error_pattern, num_lines)
        if has_error:
            error_logs.append((idx, log_file, last_lines))
        else:
            good_files.append(idx)

    # Print results
    print("=== Existing and No Issues ===")
    for idx in sorted(good_files):
        print(f"- Index: {idx}")

    print("\n=== Missing Files ===")
    for idx in sorted(missing_files):
        print(f"- Index: {idx}")

    print("\n=== Logs with Errors ===")
    for idx, log_file, last_lines in sorted(error_logs, key=lambda x: x[0]):
        print(f"- Index: {idx}, File: {log_file}")
        print("  Last few lines:")
        for line in last_lines:
            print(f"    {line.strip()}")
        print()

if __name__ == "__main__":
    # Argument parsing with default values
    parser = ArgumentParser(description="Analyze log files for errors, missing indices, and good files.")
    parser.add_argument("--root", type=str, default="/bai-fast-vol/code/jobs-logs",
                        help="Root directory containing the log files. Default: /bai-fast-vol/code/jobs-logs")
    parser.add_argument("--keyword", type=str, default="Gendog",
                        help="Keyword to match log files (e.g., 'Gendog'). Default: Gendog")
    parser.add_argument("--max-index", type=int, required=True,
                        help="Maximum index (exclusive) for log files (e.g., for max-index=10, checks Gendog0 to Gendog9).")
    parser.add_argument("--num-lines", type=int, default=10,
                        help="Number of last lines to print for files with errors. Default: 10")
    args = parser.parse_args()

    # Run analysis
    analyze_logs(args.root, args.keyword, args.max_index, args.num_lines)
