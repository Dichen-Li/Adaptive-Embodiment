import yaml
from types import SimpleNamespace
import numpy as np
import re

# Function to convert dict to SimpleNamespace recursively
def dict_to_namespace(config_dict):
    if isinstance(config_dict, dict):
        return SimpleNamespace(**{key: dict_to_namespace(value) for key, value in config_dict.items()})
    elif isinstance(config_dict, list):
        return [dict_to_namespace(item) for item in config_dict]
    return config_dict

# Regular expression to match scientific notation
scientific_notation_pattern = re.compile(r"^-?\d+(\.\d+)?[eE][-+]?\d+$")

# Function to ensure only float-like values (including scientific notation) are converted to floats, preserving integers
def enforce_float_conversion(config_dict):
    if isinstance(config_dict, dict):
        return {key: enforce_float_conversion(value) for key, value in config_dict.items()}
    elif isinstance(config_dict, list):
        return [enforce_float_conversion(item) for item in config_dict]
    else:
        # If it's already an int, return as-is
        if isinstance(config_dict, int):
            return config_dict
        # If it's already a float, return as-is
        elif isinstance(config_dict, float):
            return config_dict
        # If it's a string that represents scientific notation, convert to float
        elif isinstance(config_dict, str) and scientific_notation_pattern.match(config_dict):
            return float(config_dict)
        # Return as-is for other types
        else:
            return config_dict

# Load YAML file and enforce the correct conversion
def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    # Ensure float-like numbers (including scientific notation in strings) are floats, but preserve integers
    config_dict = enforce_float_conversion(config_dict)
    return dict_to_namespace(config_dict)


# # Example of loading the config
# config = load_config("/home/research/github/embodiment-scaling-law/training/environments/robots/unitree_go1/config.yaml")
# import ipdb; ipdb.set_trace()
