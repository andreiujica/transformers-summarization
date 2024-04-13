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
from huggingface_hub import HfApi


# We need to use the Hugging Face Hub to save the metrics as a file
# as the Space storage resets after every build
HF_TOKEN = os.getenv('HF_TOKEN')
api = HfApi(token=HF_TOKEN)

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
        file_path = f"precomputed_metrics/{model_name['name']}_metrics.json"

        # Check if metrics file already exists
        if os.path.exists(file_path):
            logging.info(f"Metrics already exist for {model_name['name']}. Skipping...")
            continue
        
        try:
            logging.info(f"Starting evaluation for model: {model_name['name']}")
            dataset = get_dataset(only_samples=True) # TODO: Change to False after POC works
            metrics = run_evaluation_suite(model_name['name'], dataset)

            # Create a file path for the metrics within the root directory
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as outfile:
                json.dump(metrics, outfile)

            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"precomputed_metrics/{model_name['name']}_metrics.json",
                repo_id=f"andreiujica/summarization-us-patents",
                repo_type="space"
            )

            logging.info(f"Successfully saved metrics for {model_name['name']} at {file_path}: {metrics}")
        except Exception as e:
            logging.error(f"Error computing or saving metrics for {model_name['name']}: {e}")