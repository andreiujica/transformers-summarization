import gradio as gr
from utils.data import get_dataset_sample
from utils.model import load_model, generate_summary
from utils.metrics import compute_rouge_scores

# Pre-load a small set of samples for the demonstration and initial setup.
samples = get_dataset_sample(num_samples=5)

def evaluate_model(model_name, input_text):
    # Load the model and tokenizer based on user selection.
    model, tokenizer = load_model(model_name)
    
    # Generate summary for the given input text.
    generated_summary = generate_summary(model, tokenizer, input_text)
    reference_summary = samples[0]['summary']
    
    # Compute ROUGE scores for the generated summary against the reference summary.
    scores = compute_rouge_scores([generated_summary], [reference_summary])
    
    # Format the scores for display.
    formatted_scores = {key: round(value.mid.fmeasure, 4) for key, value in scores.items()}
    
    return generated_summary, reference_summary, formatted_scores

# Define the Gradio interface.
iface = gr.Interface(
    fn=evaluate_model,
    inputs=[
        gr.inputs.Dropdown(choices=['t5-small', 'bert-base-uncased', 'gpt2'], label="Select Transformer Model"),
        gr.inputs.Textbox(lines=20, placeholder="Enter patent text here or select an example...", label="Input Patent Description"),
    ],
    outputs=[
        gr.outputs.Textbox(label="Generated Summary"),
        gr.outputs.Textbox(label="Reference Summary"),
        gr.outputs.JSON(label="ROUGE Scores"),
    ],
    examples=[[model, sample['text']] for model in ['t5-small', 'bert-base-uncased', 'gpt2'] for sample in samples],
    title="Transformer Model Summarization Benchmark",
    description="This app benchmarks the out-of-the-box summarization capabilities of various transformer models using the BIGPATENT dataset."
)

if __name__ == "__main__":
    iface.launch()
