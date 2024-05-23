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

def load_example(example):
    example_index = int(example.split(" ")[1]) - 1
    selected_example = examples[example_index]
    return selected_example["description"], f"Abstract: {selected_example['abstract']}"

# Create Gradio interface
iface = gr.Interface(
    fn=summarize_text, 
    inputs=[
        gr.Dropdown(choices=["Example 1", "Example 2", "Example 3"], label="Select Example"),
        gr.Textbox(lines=5, label="Input Text", interactive=True)
    ],
    outputs=gr.Textbox(lines=5, label="Summary"),
    title="Patent Summarizer",
    description="Summarize patents using LED trained on the BigPatent dataset.",
    examples=[
        ["Example 1", examples[0]["description"]],
        ["Example 2", examples[1]["description"]],
        ["Example 3", examples[2]["description"]],
    ]
)

dropdown_iface = gr.Interface(
    fn=load_example,
    inputs=gr.Dropdown(choices=["Example 1", "Example 2", "Example 3"], label="Select Example"),
    outputs=[
        gr.Textbox(lines=5, label="Input Text", interactive=True),
        gr.Textbox(lines=5, label="Reference", interactive=False)
    ]
)

if __name__ == "__main__":
    iface.launch()
    dropdown_iface.launch()