"""
This file contains utility functions for running inference in zero-shot mode for a specific model.
It provides functions to:
- Generate a summary
"""

import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def run_inference(model_name, input_text):
    """
    Generate a zero-shot summary for the given input text.
    
    Params:
    - model_name: The name of the model to use
    - input_text: The input text to summarize

    @return:
    - summary: The generated summary
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_text = f'summarize: {input_text}'

    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
