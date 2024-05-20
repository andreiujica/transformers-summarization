import gradio as gr
import optuna
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from src.finetune import ensure_equal_chunks
from evaluate import load
from src.summarize import load_model_and_tokenizer


model, tokenizer = load_model_and_tokenizer("allenai/led-base-16384")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
dataset = load_dataset('big_patent', 'g', trust_remote_code=True)

train_dataset = dataset['train'].select(range(100))  # Use a subset for demonstration
val_dataset = dataset['validation'].select(range(20))

def preprocess_data(examples):
    descriptions = examples['description']
    abstracts = examples['abstract']
    all_inputs = []
    
    for description, abstract in zip(descriptions, abstracts):
        document_chunks, summary_sentences = ensure_equal_chunks(description, abstract)
        for chunk, summary_sentence in zip(document_chunks, summary_sentences):
            inputs = tokenizer(chunk, max_length=16384, truncation=True, padding="max_length")
            labels = tokenizer(summary_sentence, max_length=16384, truncation=True, padding="max_length")
            inputs['labels'] = labels['input_ids']
            all_inputs.append(inputs)
    
    return {
        'input_ids': [input['input_ids'] for input in all_inputs],
        'attention_mask': [input['attention_mask'] for input in all_inputs],
        'labels': [input['labels'] for input in all_inputs]
    }

train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=["description", "abstract"])
val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=["description", "abstract"])

metric = load("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions.argmax(-1)
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    rouge_scores = metric.compute(predictions=pred_str, references=label_str)
    
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
    }


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        remove_unused_columns=False,
        logging_dir='./logs',
        predict_with_generate=True,
        fp16=True,
    )
    
    # Initialize Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    
    # Return the evaluation loss
    return eval_results["eval_loss"]


def run_optuna(n_trials):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    
    best_params = None
    progress = gr.Progress(track_tqdm=True)
    for _ in progress.tqdm(range(n_trials), desc="Running Optuna trials"):
        study.optimize(lambda trial: objective(trial), n_trials=1, n_jobs=4)
    
    return best_params

def gradio_app(n_trials):
    best_params = run_optuna(n_trials)
    return best_params


iface = gr.Interface(
    fn=gradio_app,
    inputs=gr.Slider(minimum=1, maximum=50, step=1, value=20, label="Number of Trials"),
    outputs="json",
    title="Optuna Hyperparameter Optimization",
    description="Fine-tune LED on BigPatent dataset with Optuna for hyperparameter optimization."
)

if __name__ == "__main__":
    iface.launch()