# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub", "polars"]
# ///
"""
Pull the freshly-labelled sentiment dataset and emit data.json matching the
schema expected by posts/2026/agent-sentiment/index.qmd:

    [
      {
        "turn": int,
        "nTurns": int,
        "normPos": float,
        "model": str,
        "sessionId": str,
        "label": "POSITIVE" | "NEUTRAL" | "NEGATIVE",
        "sentiment": int  (+1 / 0 / -1)
      },
      ...
    ]
"""

import json
from pathlib import Path

import polars as pl
from huggingface_hub import snapshot_download

OUT_PATH = Path(
    "/Users/davanstrien/Documents/daniel/blog/posts/2026/agent-sentiment/data.json"
)
SRC = "davanstrien/agent-trace-sentiment"


def main() -> None:
    print(f"Downloading {SRC}...", flush=True)
    local_dir = snapshot_download(
        repo_id=SRC,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
    )
    df = pl.read_parquet(f"{local_dir}/data/*.parquet")
    print(f"Loaded {df.height:,} rows. Columns: {df.columns}", flush=True)

    label_to_score = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}

    records = []
    skipped = 0
    for row in df.iter_rows(named=True):
        label = row.get("sentiment_label")
        if label not in label_to_score:
            skipped += 1
            continue
        records.append(
            {
                "turn": int(row["turn"]) if row.get("turn") is not None else None,
                "nTurns": int(row["nTurns"]) if row.get("nTurns") is not None else None,
                "normPos": float(row["normPos"]) if row.get("normPos") is not None else None,
                "model": row.get("model"),
                "sessionId": row.get("session_id"),
                "label": label,
                "sentiment": label_to_score[label],
            }
        )

    print(f"Kept {len(records):,} rows, skipped {skipped} with missing/ERROR labels", flush=True)

    OUT_PATH.write_text(json.dumps(records))
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"Wrote {OUT_PATH} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
