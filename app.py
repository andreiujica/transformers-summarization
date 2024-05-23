import gradio as gr
from transformers import pipeline
from datasets import load_dataset

MODEL_NAME = "andreiujica/led-base-big-patent"
summarizer = pipeline("summarization", model=MODEL_NAME, min_length=20, max_length=128)
dataset = load_dataset("NortheasternUniversity/big_patent", "g", split="test", trust_remote_code=True, streaming=True)
tokenizer_kwargs = {'padding':True, 'truncation':True, 'return_tensors':'pt'}


examples = []
for i, example in enumerate(dataset):
    if i >= 3:
        break
    examples.append({
        "abstract": example['abstract'],
        "description": example['description']
    })

def summarize_text(text):
    return summarizer(text, **tokenizer_kwargs)[0]["summary_text"]

def load_example(example):
    example_index = int(example.split(" ")[1]) - 1
    selected_example = examples[example_index]
    return selected_example["description"], selected_example["abstract"]

dropdown_iface = gr.Interface(
    fn=load_example,
    inputs=gr.Dropdown(choices=["Example 1", "Example 2", "Example 3"], label="Select Example"),
    outputs=[
        gr.Textbox(lines=10, label="Input Text", interactive=True),
        gr.Textbox(lines=10, label="Reference Summary", interactive=False)
    ]
)

# Create Gradio interface for summarization
summarize_iface = gr.Interface(
    fn=summarize_text, 
    inputs=gr.Textbox(lines=10, label="Input Text"),
    outputs=gr.Textbox(lines=10, label="Summary"),
    title="Patent Summarizer",
    description="Summarize patents using LED trained on the BigPatent dataset."
)

# Combine the interfaces
iface = gr.TabbedInterface([dropdown_iface, summarize_iface], ["Load Example", "Summarize"])

if __name__ == "__main__":
    iface.launch()