"""
This is the main entrypoint for the Hugging Face Space. It contains the Gradio interface for the summarization benchmark.
It is also the place where we will be precomputing the metrics for all models in the config file - if needed.
"""
import yaml
import gradio as gr
from datasets import load_dataset
from evaluate import load
from tqdm.auto import tqdm
from src.summarize import load_model_and_tokenizer, summarize_via_tokenbatches
import logging
import concurrent.futures

logger = logging.getLogger(__name__)

DATA_CONFIG_FILE = 'config/dataset.yaml'

with open(DATA_CONFIG_FILE, 'r') as file:
    data_config = yaml.safe_load(file)['dataset']

MODEL_CONFIG_FILE = 'config/models.yaml'

with open(MODEL_CONFIG_FILE, 'r') as file:
    models_config = yaml.safe_load(file)['models']

def process_item(item, model, tokenizer, max_input_length):
    input_text = item['description'] 
    reference = item['abstract'] 
    summary = summarize_via_tokenbatches(input_text, model, tokenizer, batch_length=max_input_length)
    return summary, reference

# Evaluate a single model
def evaluate_model(model_name):
    for model in models_config:
        if model['name'] == model_name:
            model_info = model

    model, tokenizer = load_model_and_tokenizer(model_name)
    logger.info(f"Loaded model {model_name}")

    dataset = load_dataset(data_config['name'], data_config['category'], split='test', trust_remote_code=True, streaming=True)
    logger.info(f"Loaded dataset {data_config['name']}")

    rouge = load('rouge')
    bleu = load('sacrebleu')
    meteor = load('meteor')

    predictions, references = [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_item, item, model, tokenizer, model_info["max_input_length"]) for item in dataset]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Evaluating {model_name}"):
            summary, reference = future.result()
            predictions.append(summary)
            references.append(reference)

    # Compute metrics
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=references)
    meteor_scores = meteor.compute(predictions=predictions, references=references)

    return f"ROUGE: {rouge_scores}, BLEU: {bleu_scores['score']}, METEOR: {meteor_scores['score']}"


# Define the Gradio interface.
iface = gr.Interface(
    fn=evaluate_model,
    inputs=[
        gr.Dropdown(choices=[model['name'] for model in models_config], label="Select Transformer Model"),
    ],
    outputs=[
        gr.Text(label="Evaluation Scores"),
    ],
    title="Transformer Model Summarization Benchmark",
    description="""This app benchmarks the out-of-the-box summarization capabilities of various transformer models using the BIGPATENT dataset. Select a model and see the performance metrics. Beware it will take around 5 minutes for the metrics to be computed."""
)

if __name__ == "__main__":
    iface.launch()
