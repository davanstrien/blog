"""
Run off-the-shelf sentiment model against the same 8,949 messages, compare to
our domain-aware Qwen3 labels.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest (3-class: NEG/NEU/POS)
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers",
#   "torch",
#   "polars",
#   "huggingface-hub",
# ]
# ///

import json
from pathlib import Path

import polars as pl
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SRC_REPO = "davanstrien/agent-trace-sentiment"
OUT_PATH = Path("/tmp/roberta_labels.parquet")

device = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Device: {device}")

print("Loading source dataset...")
local_dir = snapshot_download(repo_id=SRC_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"])
df = pl.read_parquet(f"{local_dir}/data/*.parquet")
print(f"Loaded {df.height} rows")

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
model.eval()

# cardiffnlp labels: 0=negative, 1=neutral, 2=positive
id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

texts = df["content_text"].to_list()
# truncate to 512 chars, same truncation as Qwen run
texts = [(t or "")[:512] for t in texts]

print("Classifying...")
labels = []
scores_neg = []
scores_neu = []
scores_pos = []
batch = 32
with torch.inference_mode():
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        for p, pr in zip(preds, probs):
            labels.append(id2label[int(p)])
            scores_neg.append(float(pr[0]))
            scores_neu.append(float(pr[1]))
            scores_pos.append(float(pr[2]))
        if (i // batch) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(texts)}...", flush=True)

print("Done classifying. Attaching to dataframe.")
df = df.with_columns(
    pl.Series("roberta_label", labels),
    pl.Series("roberta_p_neg", scores_neg),
    pl.Series("roberta_p_neu", scores_neu),
    pl.Series("roberta_p_pos", scores_pos),
)

print("\n=== Comparison ===")
qwen_counts = df.group_by("sentiment_label").len().sort("len", descending=True)
rob_counts = df.group_by("roberta_label").len().sort("len", descending=True)
print("Qwen3 (domain-aware):")
print(qwen_counts)
print("\nRoBERTa (off-the-shelf twitter):")
print(rob_counts)

print("\n=== Confusion (rows=Qwen, cols=RoBERTa) ===")
cross = df.pivot(on="roberta_label", index="sentiment_label", aggregate_function="len")
print(cross)

print("\n=== Disagreement rate ===")
agree = df.filter(pl.col("sentiment_label") == pl.col("roberta_label")).height
disagree = df.height - agree
print(f"Agree: {agree} ({100*agree/df.height:.1f}%)")
print(f"Disagree: {disagree} ({100*disagree/df.height:.1f}%)")

print("\n=== Disagreement examples ===")
print("Qwen=NEUTRAL, RoBERTa=NEGATIVE (likely dev profanity false-positives):")
ex = df.filter((pl.col("sentiment_label") == "NEUTRAL") & (pl.col("roberta_label") == "NEGATIVE")).head(5)
for row in ex.iter_rows(named=True):
    txt = (row["content_text"] or "")[:120].replace("\n", " ")
    print(f"  - {txt!r}")

print("\nQwen=POSITIVE, RoBERTa=NEUTRAL (Qwen catches affirmation RoBERTa misses):")
ex = df.filter((pl.col("sentiment_label") == "POSITIVE") & (pl.col("roberta_label") == "NEUTRAL")).head(5)
for row in ex.iter_rows(named=True):
    txt = (row["content_text"] or "")[:120].replace("\n", " ")
    print(f"  - {txt!r}")

df.write_parquet(OUT_PATH)
print(f"\nWrote {OUT_PATH}")
