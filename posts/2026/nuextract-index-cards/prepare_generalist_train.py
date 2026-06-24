# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0",
#   "huggingface-hub>=0.25",
#   "pillow>=10.4",
# ]
# ///
"""Build the multi-collection generalist SFT set: Teklia (flat) + NLS (nested) under ONE model.

Each example carries its OWN schema-conditional prompt (kie_score.build_user_text(schema)) + target
JSON, so the model learns to follow whatever schema is in the prompt. Output columns:
{image, prompt, target, collection}. Mix → davanstrien/cards-generalist-train (private).

Usage: uv run prepare_generalist_train.py --repo davanstrien/cards-generalist-train
"""

from __future__ import annotations

import argparse
import json

from datasets import concatenate_datasets, load_dataset

import kie_score as ks

# (dataset, schema_file, target_column, collection_name)
SOURCES = [
    ("davanstrien/teklia-nuextract3-flat-train", "flat_schema.json", "target", "teklia"),
    ("davanstrien/nls-cards-silver", "nls_schema.json", "extraction", "nls"),
    ("davanstrien/southborough-silver", "southborough_schema.json", "extraction", "south"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    parts = []
    for repo, schema_file, tcol, name in SOURCES:
        prompt = ks.build_user_text(json.load(open(schema_file, encoding="utf-8")))
        d = load_dataset(repo, split="train")
        d = d.map(lambda ex, p=prompt, t=tcol, n=name: {"prompt": p, "target": ex[t], "collection": n},
                  remove_columns=[c for c in d.column_names if c not in ("image",)])
        print(f"  {name}: {len(d)}")
        parts.append(d)

    mix = concatenate_datasets(parts).shuffle(seed=42)
    mix.push_to_hub(args.repo, private=True)
    print(f"published {len(mix)} ({len(parts)} collections) -> {args.repo} (private)")


if __name__ == "__main__":
    main()
