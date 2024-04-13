"""
This is the main entrypoint for the Hugging Face Space. It contains the Gradio interface for the summarization benchmark.
It is also the place where we will be precomputing the metrics for all models in the config file - if needed.
"""

import logging
import yaml
import gradio as gr
from datasets import Dataset
from src.precompute import precompute_average_metrics
from src.dataset import get_dataset
from src.evaluation import run_evaluation_suite

DATA_CONFIG_FILE = 'config/dataset.yaml'

with open(DATA_CONFIG_FILE, 'r') as file:
    data_config = yaml.safe_load(file)['dataset']

MODEL_CONFIG_FILE = 'config/models.yaml'

with open(MODEL_CONFIG_FILE, 'r') as file:
    models_config = yaml.safe_load(file)['models']
    models_array = [model['name'] for model in models_config]


# Compute the average metrics for the entire dataset if the precomputed metrics do not exist.
precompute_average_metrics()

def run_demo(model_name):
    """
    Run the summarization benchmark for a single model and five samples.

    Params:
    - model_name: The name of the model to use

    @return:
    - evaluation_scores: The evaluation JSON scores - ROUGE, BLEU, and METEOR
    # TODO: Fix demo dataset not working
    """

    dataset = get_dataset(only_samples=True)
    evaluation_scores = run_evaluation_suite(model_name, dataset)

    return evaluation_scores

# Define the Gradio interface.
iface = gr.Interface(
    fn=run_demo,
    inputs=[
        gr.Dropdown(choices=models_array, label="Select Transformer Model"),
    ],
    outputs=[
        gr.JSON(label="Evaluation Scores"),
    ],
    title="Transformer Model Summarization Benchmark",
    description="""This app benchmarks the out-of-the-box summarization capabilities of various transformer models using the BIGPATENT dataset. Select a model and performance metrics."""
)

if __name__ == "__main__":
    iface.launch()
