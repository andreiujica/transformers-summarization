"""
This file contains utility functions for running the summarizationEvaluator.
It provides functions to:
- Compute the ROUGE, BLEU, and METEOR scores
"""

import yaml
from evaluate import evaluator, combine

DATA_CONFIG_FILE = 'config/dataset.yaml'

with open(DATA_CONFIG_FILE, 'r') as file:
    data_config = yaml.safe_load(file)['dataset']

def run_evaluation_suite(model_name, dataset):
    """
    Runs the summarizationEvaluator to compute the ROUGE, BLEU, and METEOR scores
    for a single model on an entire dataset.

    Params:
    - model_name: The name of the model to evaluate
    - dataset: The dataset object
    """

    summarization_evaluator = evaluator('summarization')

    return summarization_evaluator.compute(
        model_or_pipeline=model_name,
        data=dataset,
        metric=combine(['rouge', 'sacrebleu', 'meteor']),
        input_column=data_config['input_column'],
        label_column=data_config['summary_column'],
        strategy='bootstrap',
    )

