#!/usr/bin/env uv run --with transformers,datasets,accelerate,scikit-learn,evaluate,huggingface_hub
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers",
#   "datasets",
#   "accelerate",
#   "scikit-learn",
#   "evaluate",
#   "huggingface_hub",
#   "torch",
# ]
# ///

import os
import evaluate
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

MODEL_ID = "roberta-base"
DATASET_ID = "biglam/on_the_books"
OUTPUT_REPO = "davanstrien/jim-crow-laws-pi-kimi"
TEXT_COLUMN = "section_text"
LABEL_COLUMN = "jim_crow"

# Load dataset
ds = load_dataset(DATASET_ID, split="train")

# Stratified split
train_idx, val_idx = train_test_split(
    range(len(ds)),
    test_size=0.2,
    stratify=[ex[LABEL_COLUMN] for ex in ds],
    random_state=42,
)
train_ds = ds.select(train_idx)
val_ds = ds.select(val_idx)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess(examples):
    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        max_length=512,
    )

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

# Rename label column for Trainer
train_ds = train_ds.rename_column(LABEL_COLUMN, "labels")
val_ds = val_ds.rename_column(LABEL_COLUMN, "labels")

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    id2label={0: "no_jim_crow", 1: "jim_crow"},
    label2id={"no_jim_crow": 0, "jim_crow": 1},
)

# Metrics
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels, average="binary")

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    report_to="none",
    seed=42,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# Push final model + tokenizer
trainer.push_to_hub()
print(f"Model pushed to {OUTPUT_REPO}")
