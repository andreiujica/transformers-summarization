import gradio as gr
from utils.data import get_dataset_sample
from utils.model import load_model, generate_summary
from utils.metrics import compute_rouge_scores

# Pre-load a small set of samples for the demonstration and initial setup. 
samples = get_dataset_sample(num_samples=5)

def evaluate_model(model_name, sample_index):
    # Load the model and tokenizer based on user selection.
    model, tokenizer = load_model(model_name)
    
    input_text = samples[sample_index]['text']
    reference_summary = samples[sample_index]['summary']
    
    # Generate summary for the given input text.
    generated_summary = generate_summary(model, tokenizer, input_text)
    
    # Compute ROUGE scores for the generated summary against the reference summary.
    scores = compute_rouge_scores([generated_summary], [reference_summary])
    
    # Format the scores for display.
    formatted_scores = {key: round(value.mid.fmeasure, 4) for key, value in scores.items()}
    
    return input_text, generated_summary, reference_summary, formatted_scores

# Define the Gradio interface.
iface = gr.Interface(
    fn=evaluate_model,
    inputs=[
        gr.Dropdown(choices=["t5-small", "facebook/bart-large", "google/pegasus-large"], label="Select Transformer Model"),
        gr.Dropdown(choices=list(range(len(samples))), label="Sample Index"),
    ],
    outputs=[
        gr.Textbox(label="Input Patent Description"),
        gr.Textbox(label="Generated Summary"),
        gr.Textbox(label="Reference Summary"),
        gr.JSON(label="ROUGE Scores"),
    ],
    title="Transformer Model Summarization Benchmark",
    description="This app benchmarks the out-of-the-box summarization capabilities of various transformer models using the BIGPATENT dataset. Select a sample index to view its input and summaries."
)

if __name__ == "__main__":
    iface.launch()
