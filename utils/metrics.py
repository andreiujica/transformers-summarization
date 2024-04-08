import evaluate
import time
from utils.model import generate_summary, load_model

def compute_metrics(predictions, references):
    # Initialize metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    
    # Compute scores
    rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bleu_scores = bleu.compute(predictions=predictions, references=references)
    meteor_scores = meteor.compute(predictions=predictions, references=references)
    
    # Simplify ROUGE, BLEU and METEOR scores for display
    formatted_rouge_scores = round(rouge_scores["rougeL"].mid.fmeasure, 4)
    formatted_bleu_score = round(bleu_scores["score"], 4)
    formatted_meteor_score = round(meteor_scores["score"], 4)
    
    return formatted_rouge_scores, formatted_bleu_score, formatted_meteor_score

def measure_inference_time(model, tokenizer, input_text):
    start_time = time.time()
    summary = generate_summary(model, tokenizer, input_text)
    end_time = time.time()
    inference_time = round(end_time - start_time, 4)
    return summary, inference_time

def measure_load_time(model_name):
    start_time = time.time()
    model, tokenizer = load_model(model_name)
    end_time = time.time()
    load_time = round(end_time - start_time, 4)
    return model, tokenizer, load_time
