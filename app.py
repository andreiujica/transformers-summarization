import gradio as gr
from transformers import pipeline
from datasets import load_dataset

MODEL_NAME = "andreiujica/led-base-big-patent"
summarizer = pipeline("summarization", model=MODEL_NAME, min_length=20, max_length=128)
dataset = load_dataset("NortheasternUniversity/big_patent", "g", split="test", trust_remote_code=True, streaming=True)

examples = []
for i, example in enumerate(dataset):
    if i >= 3:
        break
    examples.append({
        "abstract": example['abstract'],
        "description": example['description']
    })

def summarize_text(text):
    return summarizer(text)[0]["summary_text"]

# Create Gradio interface
iface = gr.Interface(
    fn=summarize_text, 
    inputs=[
        gr.Textbox(lines=5, label="Input Text", interactive=True)
    ],
    outputs=gr.Textbox(lines=5, label="Summary"),
    title="Patent Summarizer",
    description="Summarize patents using LED trained on the BigPatent dataset.",
    examples=[
        [examples[0]["description"]],
        [examples[1]["description"]],
        [examples[2]["description"]],
    ]
)

if __name__ == "__main__":
    iface.launch()