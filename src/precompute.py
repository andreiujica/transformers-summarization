"""
This file contains the method that will be run to compute the average metrics for a larger number of samples.
It will only be run if the precomputed metrics file does not exist.
"""

import os
import json
import yaml
import logging
from src.evaluation import run_evaluation_suite
from src.dataset import get_dataset
from huggingface_hub import Repository, HfFolder

MODELS_CONFIG_FILE = 'config/models.yaml'
HF_TOKEN = os.getenv('HF_TOKEN')

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

    # We need to use the Hugging Face Hub to save the metrics as a file
    # as the Space storage resets after every build
    repo_path = "./" 
    repo = Repository(local_dir=repo_path, use_auth_token=True)

    for model_name in models_config:
        try:
            logging.info(f"Starting evaluation for model: {model_name['name']}")
            dataset = get_dataset(only_samples=True) # TODO: Change to False after POC works
            metrics = run_evaluation_suite(model_name['name'], dataset)

            # Create a file path for the metrics within the root directory
            file_path = f"precomputed_metrics/{model_name['name']}_metrics.json"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as outfile:
                json.dump(metrics, outfile)

            # Add, commit, and push the file to the repository
            repo.git_add(file_path)
            repo.git_commit(f"feat(precompute): Add metrics for {model_name['name']}")
            repo.git_push()

            logging.info(f"Successfully saved metrics for {model_name['name']} at {file_path}: {metrics}")
        except Exception as e:
            logging.error(f"Error computing or saving metrics for {model_name['name']}: {e}")