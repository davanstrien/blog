# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0",
#   "huggingface-hub>=0.25",
#   "pillow>=10.4",
#   "requests>=2.32",
# ]
# ///
"""Build the Teklia *train* split as a prepared dataset for SFT + GRPO.

Mirrors teklia_prepare.py (IIIF fetch + XML GT), but for the 436-card train split and
with a flat-JSON **target** column derived from the GT via kie_score (so SFT supervises
`image -> flat JSON` and GRPO can score against `gt_xml`). Blank cards (no GT) are skipped.

The dataset is deliberately *simple* — `record_id, image, target, gt_xml` — and the train
scripts build the chat messages on the fly (one prompt builder, kie_score.build_user_text),
so SFT, GRPO and eval all share the exact same prompt framing.

Usage:
    uv run prepare_train.py --repo davanstrien/teklia-nuextract3-flat-train [--num-samples 8] [--split train]
"""

from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from datasets import Dataset, Features, Image, Value, load_dataset
from huggingface_hub import HfApi
from PIL import Image as PILImage

import kie_score as ks

SOURCE = "Teklia/DAI-CReTDHI-IndexCards-KIE"


def fetch_jpeg(url: str, attempts: int = 4) -> bytes:
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=(10, 60))
            r.raise_for_status()
            im = PILImage.open(io.BytesIO(r.content)).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            return buf.getvalue()
        except Exception:  # noqa: BLE001
            time.sleep(3 * (i + 1))
    return b""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-samples", type=int, default=None, help="Limit (for smoke tests)")
    ap.add_argument("--key-map", default="key_map.json")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    args = ap.parse_args()

    key_map = json.load(open(args.key_map, encoding="utf-8"))
    ds = load_dataset(SOURCE, split=args.split)
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"loaded {len(ds)} rows from {SOURCE} [{args.split}]")

    # Build targets first; skip blank-GT cards (empty target = no positive signal).
    keep = []
    for r in ds:
        target = ks.gt_xml_to_flat(r["text"], key_map)
        if target:
            keep.append((r, target))
    print(f"{len(keep)}/{len(ds)} cards have non-empty GT (blanks skipped)")

    # Concurrent IIIF fetch.
    urls = [r["record_url"] for r, _ in keep]
    with ThreadPoolExecutor(max_workers=12) as ex:
        images = list(ex.map(fetch_jpeg, urls))

    rows = []
    for (r, target), img in zip(keep, images):
        if not img:
            print(f"  ! {r['record_id']}: image fetch failed, skipping")
            continue
        rows.append({
            "record_id": r["record_id"],
            "image": {"bytes": img, "path": f"{r['record_id']}.jpg"},
            "target": json.dumps(target, ensure_ascii=False),
            "gt_xml": r["text"],
        })
    print(f"prepared {len(rows)} rows with images")

    features = Features({
        "record_id": Value("string"),
        "image": Image(),
        "target": Value("string"),
        "gt_xml": Value("string"),
    })
    Dataset.from_list(rows, features=features).push_to_hub(args.repo, private=args.private)

    readme = f"""---
license: mit
tags: [kie, nuextract, sft, grpo, archives, french]
---

# Teklia NuExtract-3 flat train set (private)

{len(rows)} cards from the **`{args.split}`** split of
[`{SOURCE}`](https://huggingface.co/datasets/{SOURCE}) (Archives of Tours, MIT), prepared for
SFT + GRPO domain-adaptation of [NuExtract-3](https://huggingface.co/numind/NuExtract3).

Columns: `record_id` · `image` (IIIF JPEG) · `target` (flat-by-role JSON, the SFT label) ·
`gt_xml` (nested XML GT, the GRPO reward reference). Blank-GT cards skipped. Schema +
scorer: the `nuextract-index-cards` blog-post folder.
"""
    HfApi().upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="dataset")
    print(f"published {len(rows)} rows -> {args.repo}")


if __name__ == "__main__":
    main()
