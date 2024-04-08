"""
This file contains the method that will be run to compute the average metrics for a larger number of samples.
It will only be run if the precomputed metrics file does not exist.
"""

import json
import yaml
from src.evaluation import run_evaluation_suite
from src.dataset import get_dataset

MODELS_CONFIG_FILE = 'config/models.yaml'

with open(MODELS_CONFIG_FILE, 'r') as file:
    models_config = yaml.safe_load(file)['models']

def precompute_average_metrics():
    """
    Compute the average metrics for all models in our config file.

    Params:
    - None

    @return:
    - None, but the metrics are saved to a file
    """
    for model_name in models_config['models']:
        dataset = get_dataset(only_samples=False)
        metrics = run_evaluation_suite(model_name['name'], dataset)

        with open(f"../precomputed_metrics/{model_name}_metrics.json", "w") as outfile:
            json.dump(metrics, outfile)
