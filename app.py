import gradio as gr
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from src.finetune import ensure_equal_chunks
from evaluate import load
from src.summarize import load_model_and_tokenizer
import numpy as np
import logging
import os


logger = logging.getLogger(__name__)
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "allenai/led-base-16384"
_, tokenizer = load_model_and_tokenizer(MODEL_NAME)

def model_init(trial=None):
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_init())
dataset = load_dataset('big_patent', 'g', trust_remote_code=True)

train_dataset = dataset['train'].select(range(100))
val_dataset = dataset['validation'].select(range(20))

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

LEARNING_RATE = 4.892476e-5
NUM_TRAIN_EPOCHS = 3

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_total_limit=1,
    logging_dir='./logs',
    predict_with_generate=True,
    fp16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    push_to_hub=True,
    push_to_hub_model_id="andreiujica/led-base-big-patent",
    push_to_hub_token=os.getenv("HF_TOKEN")
)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

def gradio_interface():
    trainer.train()
    trainer.push_to_hub()

    logger.info(f"Success! Your model finished training, here are the evaluation metrics: {trainer.evaluate()}")

    return trainer.evaluate()

"""
TODO:
1. Find length distribution so that you know how much to pad the shit  --DONE
2. Look into how accelerate and multiple gpu training works --DONE
3. Do optuna in order to find the parameters --DONE
4. Fine-tune small
5. Run inference to check if it works
6. Fine-tune large
6. Run rouge on that inference to see what's up
7. push to hub
"""

iface = gr.Interface(
    fn=gradio_interface,
    inputs=None,
    outputs=gr.JSON(label="Finetuning Evaluation Results"),
    title="Finetuning for BigPatent Summarization",
    description="Perform Finetuning using Hugging Face Transformers.",
)

if __name__ == "__main__":
    iface.launch()