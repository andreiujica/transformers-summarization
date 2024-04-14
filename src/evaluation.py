"""
This file contains utility functions for running the summarizationEvaluator.
It provides functions to:
- Compute the ROUGE, BLEU, and METEOR scores
"""

import yaml
from evaluate import load
from transformers import pipeline, AutoTokenizer
import logging

DATA_CONFIG_FILE = 'config/dataset.yaml'
CONTEXT_LENGTH = 512

with open(DATA_CONFIG_FILE, 'r') as file:
    data_config = yaml.safe_load(file)['dataset']

def compute_average_metrics(metrics):
    """
    Compute the average metrics for a list of metrics.

    Params:
    - metrics: A list of metrics to average

    @return:
    - average_scores: The average scores for the metrics
    """
    average_scores = {}
    for metric_name in ['rouge', 'bleu', 'meteor']:
        total = {key: sum([metric[metric_name][key] for metric in metrics]) for key in metrics[0][metric_name]}
        average = {key: total[key] / len(metrics) for key in total}
        average_scores[metric_name] = average

    return average_scores

def split_into_chunks(model_name, text):
    """
    Splits the text into manageable chunks with overlap.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_text = tokenizer.encode(text)
    length = len(tokenized_text)
    size = CONTEXT_LENGTH - 50 
    chunks = [tokenizer.decode(tokenized_text[i:i + size]) for i in range(0, length, size)]
    return chunks

def run_evaluation_suite(model_name, dataset):
    """
    Runs the summarizationEvaluator to compute the ROUGE, BLEU, and METEOR scores
    for a single model on an entire dataset.

    Params:
    - model_name: The name of the model to evaluate
    - dataset: The dataset object
    
    #TODO: 2. Make this run few shot evaluation as well using a custom pipeline
    #TODO: 2.1. Fix the context length thing for t5
    """
    
    # Load the summarization model
    summarizer = pipeline("summarization", model=model_name)

    # Load metrics
    rouge = load("rouge")
    sacrebleu = load("sacrebleu")
    meteor = load("meteor")

    # Prepare few-shot examples by taking the first two entries from the dataset
    few_shot_examples = ["Text: " + item[data_config['input_column']] + " Summary: " + item[data_config['summary_column']]
                         for item in dataset.select(range(2))]

    zero_shot_metrics, few_shot_metrics = [], []

    # Evaluate using zero-shot
    for example in dataset:
        input_text = example[data_config['input_column']]
        chunks = split_into_chunks(model_name, input_text)
        summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
        combined_summary = " ".join(summaries)

        zero_shot_metrics.append({
            "rouge": rouge.compute(predictions=[combined_summary], references=[example[data_config['summary_column']]]),
            "bleu": sacrebleu.compute(predictions=[combined_summary], references=[example[data_config['summary_column']]])["score"],
            "meteor": meteor.compute(predictions=[combined_summary], references=[example[data_config['summary_column']]])
        })

    # Evaluate using few-shot
    for example in dataset:
        input_text = " ".join(few_shot_examples + [example[data_config['input_column']]])
        chunks = split_into_chunks(model_name, input_text)
        summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
        combined_summary = " ".join(summaries)

        few_shot_metrics.append({
            "rouge": rouge.compute(predictions=[combined_summary], references=[example[data_config['summary_column']]]),
            "bleu": sacrebleu.compute(predictions=[combined_summary], references=[example[data_config['summary_column']]])["score"],
            "meteor": meteor.compute(predictions=[combined_summary], references=[example[data_config['summary_column']]])
        })

    # Calculate average scores across all examples
    zero_shot_results = compute_average_metrics(zero_shot_metrics)
    few_shot_results = compute_average_metrics(few_shot_metrics)

    # Prepare the results in JSON format
    results_json = {
        "zero-shot": zero_shot_results,
        "few-shot": few_shot_results
    }

    return results_json

