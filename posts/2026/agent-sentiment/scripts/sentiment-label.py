# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "flashinfer-python",
#     "huggingface-hub[hf_transfer]",
#     "hf-xet>= 1.1.7",
#     "polars",
#     "torch",
#     "transformers",
#     "vllm>=0.8.5",
# ]
#
# ///
"""
Classify sentiment of coding-agent user messages using vLLM.

Reads a dataset of user messages, classifies each as POSITIVE/NEUTRAL/NEGATIVE
with a domain-aware prompt, and saves results back to the Hub.

Default model: google/gemma-4-26B-A4B-it (small, fast, proven).
Override with --model-id (e.g. Qwen/Qwen3.6-35B-A3B).
Structured output via StructuredOutputsParams ensures valid JSON.

Example (HF Jobs):
    hf jobs uv run \
        --flavor a100-large \
        --secrets HF_TOKEN \
        --timeout 45m \
        --detach \
        sentiment-label.py \
        davanstrien/agent-trace-user-messages \
        davanstrien/agent-trace-sentiment
"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Import vllm FIRST so it initializes its multiprocessing + CUDA correctly
# before any torch-touching imports (transformers, datasets) create a CUDA
# context in the main process.
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import argparse
import json
import logging
import sys
from typing import Optional

import polars as pl
from datasets import Dataset
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Classify user messages from coding-agent sessions (developer talking to AI coding assistant).

Return JSON with two fields:
- "label": one of "POSITIVE", "NEUTRAL", or "NEGATIVE"
- "reason": one sentence explaining why

Domain rules:
- Dev profanity is casual ("kill that shit" = "remove code") = NEUTRAL
- Short commands ("do it", "commit and push") are approvals = NEUTRAL
- Status reports ("ci failed") = NEUTRAL
- Frustration with agent output quality = NEGATIVE
- Satisfaction, excitement about progress = POSITIVE"""

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["POSITIVE", "NEUTRAL", "NEGATIVE"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
}


def main(
    src_dataset: str,
    output_dataset: str,
    model_id: str = "google/gemma-4-26B-A4B-it",
    text_column: str = "content_text",
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = 4096,
    hf_token: Optional[str] = None,
):
    # Auth
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    # Load model — let vLLM handle GPU init; single-GPU offline inference
    logger.info(f"Loading model: {model_id}")
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Structured output — enforces valid JSON matching our schema
    structured_params = StructuredOutputsParams(json=SENTIMENT_SCHEMA)
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=200,
        structured_outputs=structured_params,
    )

    # Load dataset — bypass datasets.load_dataset to avoid stale README YAML cast
    logger.info(f"Downloading dataset: {src_dataset}")
    local_dir = snapshot_download(
        repo_id=src_dataset,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        token=token,
    )
    df = pl.read_parquet(f"{local_dir}/data/*.parquet")
    logger.info(f"Loaded {df.height:,} rows, cols: {df.columns}")

    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found. Available: {df.columns}")
        sys.exit(1)

    dataset = Dataset.from_polars(df)

    # Build prompts — thinking disabled for deterministic short output
    logger.info("Building prompts...")
    prompts = []
    for example in dataset:
        msg = example[text_column][:512]  # truncate long messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)

    # Generate — vLLM handles batching, structured output enforces schema
    logger.info(f"Classifying {len(prompts):,} messages...")
    outputs = llm.generate(prompts, sampling_params)

    # Parse responses — should be valid JSON thanks to structured output
    logger.info("Parsing responses...")
    labels = []
    reasons = []
    parse_errors = 0

    for output in outputs:
        text = output.outputs[0].text.strip()
        try:
            parsed = json.loads(text)
            labels.append(parsed.get("label", "ERROR"))
            reasons.append(parsed.get("reason", ""))
        except (json.JSONDecodeError, KeyError):
            parse_errors += 1
            labels.append("ERROR")
            reasons.append(text[:100])

    logger.info(f"Parse errors: {parse_errors}/{len(outputs)}")

    # Add columns
    dataset = dataset.add_column("sentiment_label", labels)
    dataset = dataset.add_column("sentiment_reason", reasons)

    # Log distribution
    from collections import Counter
    dist = Counter(labels)
    for label, count in dist.most_common():
        pct = 100 * count / len(labels)
        logger.info(f"  {label}: {count:,} ({pct:.1f}%)")

    # Write results to mounted volume (if available) for live inspection
    volume_path = "/mnt/results"
    if os.path.isdir("/mnt"):
        os.makedirs(volume_path, exist_ok=True)
        dataset.to_parquet(f"{volume_path}/sentiment_results.parquet")
        with open(f"{volume_path}/summary.json", "w") as f:
            json.dump({"total": len(labels), "distribution": dict(dist), "parse_errors": parse_errors}, f, indent=2)
        logger.info(f"Wrote results to {volume_path}/")

    # Push to Hub
    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, token=token)
    logger.info(f"Done! https://huggingface.co/datasets/{output_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment-label coding agent user messages with vLLM")
    parser.add_argument("src_dataset", help="Input dataset on HF Hub")
    parser.add_argument("output_dataset", help="Output dataset on HF Hub")
    parser.add_argument("--model-id", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--text-column", default="content_text")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--hf-token", type=str)
    args = parser.parse_args()

    main(
        src_dataset=args.src_dataset,
        output_dataset=args.output_dataset,
        model_id=args.model_id,
        text_column=args.text_column,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        hf_token=args.hf_token,
    )
