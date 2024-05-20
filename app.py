import gradio as gr
import optuna
from transformers import LEDForConditionalGeneration, LEDTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from src.finetune import StreamedDataset
from evaluate import load
from src.summarize import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer("allenai/led-base-16384")
metric = load("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Group by document ID
    doc_results = {}
    for pred, label, doc_id in zip(pred_str, labels_str, pred.predictions.doc_ids):
        if doc_id not in doc_results:
            doc_results[doc_id] = {"pred": [], "label": []}
        doc_results[doc_id]["pred"].append(pred)
        doc_results[doc_id]["label"].append(label)

    # Aggregate results per document
    aggregated_preds = [" ".join(doc_results[doc_id]["pred"]) for doc_id in doc_results]
    aggregated_labels = [" ".join(doc_results[doc_id]["label"]) for doc_id in doc_results]

    result = metric.compute(predictions=aggregated_preds, references=aggregated_labels)
    return result

def create_small_dataset(dataset, num_samples):
    return dataset.select(range(num_samples))

initial_sample_size = 1000
increment_size = 1000
max_sample_size = 5000 

def objective(trial, num_samples=initial_sample_size):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [1, 2, 4])
    
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=True,
    )

    full_train_dataset = load_dataset('big_patent', 'g', split='train', streaming=True)
    full_eval_dataset = load_dataset('big_patent', 'g', split='validation', streaming=True)
    
    small_train_dataset = create_small_dataset(full_train_dataset, num_samples)
    small_eval_dataset = create_small_dataset(full_eval_dataset, num_samples)

    train_dataset = StreamedDataset(small_train_dataset, tokenizer, chunk_size=16000)
    eval_dataset = StreamedDataset(small_eval_dataset, tokenizer, chunk_size=16000)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics  # Custom compute_metrics function
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result['eval_loss']


def run_optuna(n_trials):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    
    best_params = None
    num_samples = initial_sample_size
    while num_samples <= max_sample_size:
        with gr.Progress(track_tqdm=True, label=f"Hyperparameter Optimization Progress with {num_samples} samples") as progress:
            study.optimize(lambda trial: objective(trial, num_samples), n_trials=n_trials, n_jobs=4)
        best_params = study.best_params
        num_samples += increment_size
    
    return best_params

def gradio_app(n_trials):
    best_params = run_optuna(n_trials)
    return best_params


iface = gr.Interface(
    fn=gradio_app,
    inputs=gr.inputs.Slider(minimum=1, maximum=50, step=1, default=20, label="Number of Trials"),
    outputs="json",
    title="Optuna Hyperparameter Optimization",
    description="Fine-tune LED on BigPatent dataset with Optuna for hyperparameter optimization."
)

if __name__ == "__main__":
    iface.launch()