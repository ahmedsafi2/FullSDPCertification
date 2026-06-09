import yaml
import torch
import os
from fastsdp_tools.utils import get_project_path


def load_dataset(yaml_file):
    """
    Load dataset from a YAML file.
    Args:
        yaml_file (str): Path to the YAML file.
    Returns:
        dict: Dictionary containing the loaded dataset.
    """
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
        print("config : ", config)

        path = get_project_path(config["data"]["path"].replace("\\", "/"))
        try:
            dataset = torch.load(path, weights_only=False)
        except TypeError:
            dataset = torch.load(path)

    return dataset["dataset"]
