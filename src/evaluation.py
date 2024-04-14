import yaml
from evaluate import load
from transformers import AutoTokenizer, pipeline
import concurrent.futures

DATA_CONFIG_FILE = 'config/dataset.yaml'
CONTEXT_LENGTH = 512

with open(DATA_CONFIG_FILE, 'r') as file:
    data_config = yaml.safe_load(file)['dataset']

metrics = {'rouge': load('rouge'), 'sacrebleu': load('sacrebleu'), 'meteor': load('meteor')}

def compute_average_metrics(metrics):
    average_scores = {metric_name: {key: sum(metric[metric_name][key] for metric in metrics) / len(metrics)
                                    for key in metrics[0][metric_name]}
                      for metric_name in ['rouge', 'bleu', 'meteor']}
    return average_scores

def split_into_chunks(tokenizer, text, overlap=50):
    tokenized_text = tokenizer.encode(text)
    size = CONTEXT_LENGTH - overlap
    chunks = [tokenized_text[i:i + size] for i in range(0, len(tokenized_text), size)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def summarize_and_evaluate(model, tokenizer, example, few_shot_prefix=""):
    input_text = few_shot_prefix + example[data_config['input_column']]
    reference = example[data_config['summary_column']]
    chunks = split_into_chunks(tokenizer, input_text)
    summaries = [model(chunk)[0]['summary_text'] for chunk in chunks]
    combined_summary = " ".join(summaries)
    return {
        "rouge": metrics['rouge'].compute(predictions=[combined_summary], references=[reference]),
        "bleu": {"score": metrics['sacrebleu'].compute(predictions=[combined_summary], references=[reference])["score"]},
        "meteor": metrics['meteor'].compute(predictions=[combined_summary], references=[reference])
    }

def run_evaluation_suite(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("summarization", model=model_name)
    few_shot_prefix = " ".join(["Text: " + item[data_config['input_column']] + " Summary: " + item[data_config['summary_column']]
                                for item in dataset.select(range(2))]) + " "

    zero_shot_metrics, few_shot_metrics = [], []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures_zero_shot = [executor.submit(summarize_and_evaluate, model, tokenizer, example) for example in dataset]
        futures_few_shot = [executor.submit(summarize_and_evaluate, model, tokenizer, example, few_shot_prefix) for example in dataset]

        zero_shot_metrics = [future.result() for future in futures_zero_shot]
        few_shot_metrics = [future.result() for future in futures_few_shot]

    zero_shot_results = compute_average_metrics(zero_shot_metrics)
    few_shot_results = compute_average_metrics(few_shot_metrics)

    results_json = {
        "zero-shot": zero_shot_results,
        "few-shot": few_shot_results
    }

    return results_json
