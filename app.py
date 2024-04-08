import gradio as gr
from utils.data import get_dataset_sample
from utils.model import load_model, generate_summary, load_model_names
from utils.metrics import measure_load_time, measure_inference_time, compute_metrics

# Pre-load a small set of samples for the demonstration and initial setup. 
samples = get_dataset_sample(num_samples=5)
model_names = load_model_names()

def evaluate_model(model_name, sample_index, summarization_mode):
    model, tokenizer, load_time = measure_load_time(model_name)
    
    input_text = samples[sample_index]['text']
    reference_summary = samples[sample_index]['summary']
    multi_shot_examples = [(sample['text'], sample['summary']) for sample in samples[:2]]
    
    if summarization_mode == 'Multi-Shot':
        generated_summary, inference_time = measure_inference_time(model, tokenizer, input_text, examples=multi_shot_examples, mode='multi-shot')
    else:
        generated_summary, inference_time = measure_inference_time(model, tokenizer, input_text)
    
    # Compute quality metrics
    rouge_scores, bleu_score, meteor_score = compute_metrics([generated_summary], [reference_summary])
    
    return {
        "Input Text": input_text,
        "Generated Summary": generated_summary,
        "Reference Summary": reference_summary,
        "ROUGE Scores": rouge_scores,
        "BLEU Score": bleu_score,
        "METEOR Score": meteor_score,
        "Model Load Time (seconds)": load_time,
        "Inference Time (seconds)": inference_time
    }

# Define the Gradio interface.
iface = gr.Interface(
    fn=evaluate_model,
    inputs=[
        gr.Dropdown(choices=model_names, label="Select Transformer Model"),
        gr.Dropdown(choices=list(range(len(samples))), label="Sample Index"),
        gr.Radio(choices=["One-Shot", "Multi-Shot"], label="Summarization Mode")
    ],
    outputs=[
        gr.Textbox(label="Input Patent Description"),
        gr.Textbox(label="Generated Summary"),
        gr.Textbox(label="Reference Summary"),
        gr.JSON(label="ROUGE Scores"),
        gr.Number(label="BLEU Score"),
        gr.Number(label="METEOR Score"),
        gr.Number(label="Model Load Time (seconds)"),
        gr.Number(label="Inference Time (seconds)")
    ],
    title="Transformer Model Summarization Benchmark",
    description="""This app benchmarks the out-of-the-box summarization capabilities of various transformer models using the BIGPATENT dataset, including one-shot and multi-shot summarization modes. Select a model, sample index, and summarization mode to view its input, summaries, and performance metrics."""
)

if __name__ == "__main__":
    iface.launch()
