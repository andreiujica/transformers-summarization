import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_names():
    with open('models_config.yaml', 'r') as file:
        models_config = yaml.safe_load(file)
    return [model['name'] for model in models_config['models']]

def load_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_summary(model, tokenizer, input_text, examples=None, mode='one-shot'):
    """
    Generate a summary for the given input text.
    
    :param model: The loaded model object.
    :param tokenizer: The loaded tokenizer object.
    :param input_text: The text to summarize.
    :param examples: A list of tuples (example_text, example_summary) for multi-shot learning.
    :param mode: 'one-shot' or 'multi-shot'. If 'multi-shot', examples must be provided.
    :return: The generated summary.
    """
    prompt_text = ""
    
    if mode == 'multi-shot' and examples is not None:
        # Construct a prompt with multiple examples
        for example_text, example_summary in examples:
            prompt_text += f"Text: {example_text}\nSummary: {example_summary}\n\n"
        prompt_text += f"Text: {input_text}\nSummary:"
    else:
        prompt_text = f"summarize: {input_text}"
    
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary