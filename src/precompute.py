"""
This file contains the method that will be run to compute the average metrics for a larger number of samples.
It will only be run if the precomputed metrics file does not exist.
"""

import json
import yaml
import logging
from src.evaluation import run_evaluation_suite
from src.dataset import get_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    for model_name in models_config:
        try:
            logging.info(f"Starting evaluation for model: {model_name['name']}")
            dataset = get_dataset(only_samples=False)
            logging.info(f"Loaded dataset for {model_name['name']}")
            metrics = run_evaluation_suite(model_name['name'], dataset)

            file_path = f"../precomputed_metrics/{model_name['name']}_metrics.json"
            with open(file_path, "w") as outfile:
                json.dump(metrics, outfile)

            logging.info(f"Successfully saved metrics for {model_name['name']} at {file_path}")
        except Exception as e:
            logging.error(f"Error computing or saving metrics for {model_name['name']}: {e}")