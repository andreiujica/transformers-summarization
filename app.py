import gradio as gr
import optuna
import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from src.finetune import ensure_equal_chunks
from evaluate import load
from src.summarize import load_model_and_tokenizer
import logging
import numpy as np


model, tokenizer = load_model_and_tokenizer("allenai/led-base-16384")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
dataset = load_dataset('big_patent', 'g', trust_remote_code=True)

train_dataset = dataset['train'].select(range(100))
val_dataset = dataset['validation'].select(range(100))

def preprocess_data(examples):
    descriptions = examples['description']
    abstracts = examples['abstract']
    all_inputs = []
    
    for description, abstract in zip(descriptions, abstracts):
        document_chunks, summary_sentences = ensure_equal_chunks(description, abstract)
        for chunk, summary_sentence in zip(document_chunks, summary_sentences):
            inputs = tokenizer(chunk, max_length=16384, truncation=True, padding="max_length")
            labels = tokenizer(summary_sentence, max_length=2048, truncation=True, padding="max_length")
            inputs['labels'] = labels['input_ids']
            all_inputs.append(inputs)
    
    return {
        'input_ids': [input['input_ids'] for input in all_inputs],
        'attention_mask': [input['attention_mask'] for input in all_inputs],
        'labels': [input['labels'] for input in all_inputs]
    }

train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=["description", "abstract"])
val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=["description", "abstract"])

rouge = load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# def objective(trial):
#     # Suggest hyperparameters
#     learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
#     num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)
#     per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4])
#     gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)
    
#     # Define training arguments
#     training_args = Seq2SeqTrainingArguments(
#         output_dir="./results",
#         evaluation_strategy="epoch",
#         learning_rate=learning_rate,
#         per_device_train_batch_size=per_device_train_batch_size,
#         num_train_epochs=num_train_epochs,
#         weight_decay=0.01,
#         save_total_limit=1,
#         remove_unused_columns=False,
#         logging_dir='./logs',
#         predict_with_generate=True,
#         fp16=True,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         ddp_find_unused_parameters=False,
#     )
    
#     # Initialize Seq2SeqTrainer
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )
    
#     # Train and evaluate
#     torch.cuda.empty_cache()
#     trainer.train()
#     eval_results = trainer.evaluate()
    
#     # Return the evaluation loss
#     return eval_results["eval_loss"]


# def run_optuna(n_trials):
#     pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
#     study = optuna.create_study(direction='minimize', pruner=pruner)
    
#     best_params = None
#     progress = gr.Progress(track_tqdm=True)
#     for _ in progress.tqdm(range(n_trials), desc="Running Optuna trials"):
#         study.optimize(lambda trial: objective(trial), n_trials=1, n_jobs=4)
    
#     return best_params

# def gradio_app(n_trials):
#     best_params = run_optuna(n_trials)
#     return best_params

import matplotlib.pyplot as plt

def compute_token_length_distribution(no_of_samples):
    
    # Compute the lengths of tokenized descriptions and summaries
    desc_lengths = [len(tokenizer.tokenize(item['description'])) for item in dataset['validation']]
    summary_lengths = [len(tokenizer.tokenize(item['abstract'])) for item in dataset['validation']]
    
    # Plot the distribution of description lengths
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(desc_lengths, bins=30, edgecolor='black')
    plt.title('Distribution of Tokenized Description Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # Plot the distribution of summary lengths
    plt.subplot(1, 2, 2)
    plt.hist(summary_lengths, bins=30, edgecolor='black')
    plt.title('Distribution of Tokenized Summary Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # Save the plot to a file
    plot_filename = "length_distribution.png"
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

# Define the Gradio interface
def gradio_interface(split):
    return compute_token_length_distribution(split)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.inputs.Textbox(lines=1, placeholder="Enter dataset split (train/test/validation)", default="train"),
    outputs=gr.outputs.Image(type="file", label="Length Distribution"),
    title="Token Length Distribution for BigPatent Descriptions and Summaries",
    description="Enter the split (train/test/validation) of the BigPatent dataset to see the distribution of tokenized description and summary lengths."
)

if __name__ == "__main__":
    iface.launch()