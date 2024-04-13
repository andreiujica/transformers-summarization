"""
This is the main entrypoint for the Hugging Face Space. It contains the Gradio interface for the summarization benchmark.
It is also the place where we will be precomputing the metrics for all models in the config file - if needed.
"""

import logging
import yaml
import gradio as gr
from datasets import Dataset
from src.precompute import precompute_average_metrics
from src.inference import run_inference
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

def run_demo(model_name, sample_idx):
    """
    Run the summarization benchmark for a single model and sample index.

    Params:
    - model_name: The name of the model to use
    - sample_idx: The index of the sample to use

    @return:
    - input_text: The input patent description
    - generated_summary: The generated summary
    - reference_summary: The reference summary
    - evaluation_scores: The evaluation JSON scores - ROUGE, BLEU, and METEOR
    # TODO: Fix demo dataset not working
    """

    dataset = get_dataset(only_samples=True)
    sample = dataset[sample_idx]
    logging.info(f"Running demo for sample: {sample}")

    input_text = sample[data_config['input_column']]
    reference_summary = sample[data_config['summary_column']]

    generated_summary = run_inference(model_name, input_text)
    evaluation_scores = run_evaluation_suite(model_name, Dataset.from_dict(sample))

    return input_text, generated_summary, reference_summary, evaluation_scores

# Define the Gradio interface.
iface = gr.Interface(
    fn=run_demo,
    inputs=[
        gr.Dropdown(choices=models_array, label="Select Transformer Model"),
        gr.Dropdown(choices=list(range(data_config["sample_size"])), label="Sample Index"),
    ],
    outputs=[
        gr.Textbox(label="Input Patent Description"),
        gr.Textbox(label="Generated Summary"),
        gr.Textbox(label="Reference Summary"),
        gr.JSON(label="Evaluation Scores"),
    ],
    title="Transformer Model Summarization Benchmark",
    description="""This app benchmarks the out-of-the-box summarization capabilities of various transformer models using the BIGPATENT dataset. Select a model, sample index, and summarization mode to view its input, summaries, and performance metrics."""
)

if __name__ == "__main__":
    iface.launch()
