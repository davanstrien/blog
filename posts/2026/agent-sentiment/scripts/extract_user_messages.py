# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "agent-traces",
#     "polars",
#     "datasets",
#     "huggingface-hub",
# ]
#
# [tool.uv.sources]
# agent-traces = { git = "https://github.com/davanstrien/agent-traces" }
# ///
"""
Extract user messages from every public `format:agent-traces` dataset on the Hub
and push a single deduped dataset for downstream sentiment labelling.

Supersedes the Jan 2026 extraction (5,891 msgs / 27 datasets). Uses the current
agent-traces parser which fixes bugs in the previous pi-trace-parser-poc.

Output schema (matches `sentiment-label.py` text_column default = "content_text"):
    id               int
    source_dataset   str     e.g. "badlogicgames/pi-mono"
    session_id       str
    turn             int     1-indexed
    nTurns           int
    normPos          float   [0, 1] normalised position
    model            str
    provider         str
    agent            str
    content_text     str     <- the user message
    timestamp        str     ISO/epoch string
    n_events         int     session-level
    n_errors         int
    n_tool_calls     int
    input_tokens_total  int
    output_tokens_total int
    cost_total_sum   float
"""

from __future__ import annotations

import os
import sys

import polars as pl
from agent_traces import TraceDataset
from datasets import Dataset
from huggingface_hub import HfApi

OUT_REPO = "davanstrien/agent-trace-user-messages"

# Datasets to skip — known outliers where the cost/benefit of including them
# for sentiment analysis is bad. jedisct1/agent-traces-swival has 8,869 JSONL
# files (>100x the median dataset) and is all security-audit material where
# "sentiment" is semantically off-prompt anyway.
SKIP_REPOS = {
    "jedisct1/agent-traces-swival",
}


def main() -> None:
    api = HfApi()

    # Grab every format:agent-traces dataset (includes re-uploads of pi-mono;
    # we dedupe on session_id across repos at the end)
    datasets = list(api.list_datasets(filter="format:agent-traces"))
    print(f"Hub reports {len(datasets)} datasets with format:agent-traces tag", flush=True)
    datasets = [d for d in datasets if d.id not in SKIP_REPOS]
    print(f"After skip-list: {len(datasets)} datasets to process", flush=True)

    per_repo: list[pl.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    for d in datasets:
        try:
            ds = TraceDataset.from_hub(d.id)
        except Exception as e:
            skipped.append((d.id, f"parse error: {type(e).__name__}: {e}"))
            continue

        um = ds.user_messages
        if um.height == 0:
            skipped.append((d.id, "no user messages"))
            continue

        um = um.with_columns(source_dataset=pl.lit(d.id))
        per_repo.append(um)
        print(f"  {d.id}: {um.height} user messages", flush=True)

    print(f"\n{len(per_repo)} datasets contributed messages, {len(skipped)} skipped", flush=True)
    for repo, reason in skipped:
        print(f"  skip {repo}: {reason}", flush=True)

    if not per_repo:
        print("No messages extracted — aborting", file=sys.stderr)
        sys.exit(1)

    combined = pl.concat(per_repo, how="diagonal_relaxed")

    # msg -> content_text (matches sentiment-label.py default)
    combined = combined.rename({"msg": "content_text"})

    # dedupe on (source_dataset, session_id, turn) to handle pi-mono re-uploads
    # keep first occurrence by source_dataset alphabetical order
    before = combined.height
    combined = combined.unique(subset=["session_id", "turn"], keep="first")
    after = combined.height
    print(f"\nDeduped {before} -> {after} rows on (session_id, turn)", flush=True)

    # drop empties (compaction summaries etc.)
    combined = combined.filter(
        pl.col("content_text").is_not_null()
        & (pl.col("content_text").str.strip_chars() != "")
    )
    print(f"Dropped empty content_text -> {combined.height} rows", flush=True)

    # stable id column
    combined = combined.with_row_index("id")
    combined = combined.with_columns(pl.col("id").cast(pl.UInt32))

    # reorder for sanity
    ordered = [
        "id",
        "source_dataset",
        "session_id",
        "turn",
        "nTurns",
        "normPos",
        "model",
        "provider",
        "agent",
        "content_text",
        "timestamp",
        "n_events",
        "n_errors",
        "n_tool_calls",
        "input_tokens_total",
        "output_tokens_total",
        "cost_total_sum",
    ]
    ordered = [c for c in ordered if c in combined.columns]
    combined = combined.select(ordered)

    print("\nFinal schema:", flush=True)
    for name, dtype in combined.schema.items():
        print(f"  {name}: {dtype}", flush=True)

    print(f"\nPushing {combined.height} messages -> {OUT_REPO}", flush=True)
    hf_ds = Dataset.from_polars(combined)
    hf_ds.push_to_hub(OUT_REPO, token=os.environ.get("HF_TOKEN"))
    print(f"Done: https://huggingface.co/datasets/{OUT_REPO}", flush=True)


if __name__ == "__main__":
    main()
