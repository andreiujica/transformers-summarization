from datasets import load_metric

def compute_rouge_scores(predictions, references):
    rouge = load_metric("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    return scores