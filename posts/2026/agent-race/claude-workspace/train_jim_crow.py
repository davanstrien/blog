# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.0",
#     "transformers>=4.48",
#     "torch>=2.4",
#     "accelerate>=1.0",
#     "scikit-learn>=1.5",
#     "huggingface-hub>=0.27",
# ]
# ///
"""Fine-tune ModernBERT-base on biglam/on_the_books for Jim Crow law detection.

Binary classification on `section_text` -> `jim_crow` (0=no_jim_crow, 1=jim_crow).
Stratified 80/20 train/eval split. Class-weighted cross-entropy to handle the
~29% positive imbalance. Pushes the trained model to the Hub.
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "answerdotai/ModernBERT-base"
DATASET_ID = "biglam/on_the_books"
PUSH_TO = "davanstrien/jim-crow-laws-claude-code"
MAX_LENGTH = 1024  # covers ~95th pct of section_text; truncates long-tail
SEED = 42


def main() -> None:
    print(f"Loading dataset {DATASET_ID}")
    raw = load_dataset(DATASET_ID, split="train")
    print(f"Total rows: {len(raw)}")

    # Stratified 80/20 split on the label
    split = raw.train_test_split(
        test_size=0.2, seed=SEED, stratify_by_column="jim_crow"
    )
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")
    print(
        "Train label dist:",
        dict(zip(*np.unique(train_ds["jim_crow"], return_counts=True))),
    )
    print(
        "Eval label dist :",
        dict(zip(*np.unique(eval_ds["jim_crow"], return_counts=True))),
    )

    # Class weights from the training split (inverse frequency)
    train_labels = np.array(train_ds["jim_crow"])
    counts = np.bincount(train_labels, minlength=2)
    n_total = counts.sum()
    class_weights = torch.tensor(
        n_total / (2.0 * counts), dtype=torch.float
    )
    print(f"Class weights: {class_weights.tolist()}")

    print(f"Loading tokenizer + model {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    label_names = raw.features["jim_crow"].names  # ["no_jim_crow", "jim_crow"]
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize(batch):
        return tokenizer(
            batch["section_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    keep_cols = ["jim_crow"]
    train_tok = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    ).rename_column("jim_crow", "labels")
    eval_tok = eval_ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in eval_ds.column_names if c not in keep_cols],
    ).rename_column("jim_crow", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1, zero_division=0
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision_jim_crow": prec,
            "recall_jim_crow": rec,
            "f1_jim_crow": f1,
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(labels, probs),
        }

    output_dir = "./jim-crow-output"

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_jim_crow",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        bf16=torch.cuda.is_available(),
        seed=SEED,
        push_to_hub=True,
        hub_model_id=PUSH_TO,
        hub_strategy="end",
        hub_private_repo=False,
    )

    class WeightedTrainer(Trainer):
        def __init__(self, *a, class_weights=None, **kw):
            super().__init__(*a, **kw)
            self._cw = class_weights

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(
                weight=self._cw.to(logits.device) if self._cw is not None else None
            )
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("Starting training")
    trainer.train()

    print("Final eval on held-out split:")
    final = trainer.evaluate()
    for k, v in final.items():
        print(f"  {k}: {v}")

    print(f"Pushing model to {PUSH_TO}")
    trainer.push_to_hub(
        commit_message="Fine-tuned ModernBERT-base on biglam/on_the_books"
    )

    # Also push tokenizer explicitly to be safe
    tokenizer.push_to_hub(PUSH_TO)
    print("Done.")


if __name__ == "__main__":
    main()
