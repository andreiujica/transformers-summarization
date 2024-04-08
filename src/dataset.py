"""
This file contains utility functions for loading and processing data.
It loads the config from the yaml file and it provides functions to:
- Load the dataset using shards
"""

import yaml
from datasets import load_dataset

DATA_CONFIG_FILE = 'config/dataset.yaml'

with open(DATA_CONFIG_FILE, 'r') as file:
    data_config = yaml.safe_load(file)['dataset']

def get_dataset(only_samples=True):
    """
    Params:
    - only_samples: If True, only a sample of the dataset is returned

    @return:
    - dataset: The dataset object
    """
    split = f'{data_config["split"]}[:{data_config["sample_size"]}]' if only_samples else data_config["split"]
    dataset = load_dataset(data_config['name'], data_config['category'], split=split, trust_remote_code=True, num_proc=8)

    return dataset
