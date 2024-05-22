import gradio as gr
import optuna
import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM, EarlyStoppingCallback
from datasets import load_dataset
from src.finetune import ensure_equal_chunks
from evaluate import load
from src.summarize import load_model_and_tokenizer
import logging
import numpy as np

MODEL_NAME = "allenai/led-base-16384"
_, tokenizer = load_model_and_tokenizer(MODEL_NAME)

def model_init(trial=None):
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda")


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_init())
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
            inputs = tokenizer(chunk, max_length=4096, truncation=True, padding="max_length")
            labels = tokenizer(summary_sentence, max_length=512, truncation=True, padding="max_length")
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

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6),
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_total_limit=1,
    logging_dir='./logs',
    predict_with_generate=True,
    fp16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    load_best_model_at_end=True,
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=1)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

def gradio_interface():
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        n_trials=10,
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics["eval_rougeLsum"],
    )

    return best_trial.hyperparameters

"""
TODO:
1. Find length distribution so that you know how much to pad the shit  --DONE
2. Look into how accelerate and multiple gpu training works --DONE
3. Do optuna in order to find the parameters
4. Fine-tune
5. Run inference to check if it works
6. Run rouge on that inference to see what's up
7. push to hub
"""

iface = gr.Interface(
    fn=gradio_interface,
    inputs=None,
    outputs=gr.JSON(label="Best Hyperparameters"),
    title="Hyperparameter Tuning for BigPatent Summarization",
    description="Perform hyperparameter tuning using Optuna and Hugging Face.",
)

if __name__ == "__main__":
    iface.launch()