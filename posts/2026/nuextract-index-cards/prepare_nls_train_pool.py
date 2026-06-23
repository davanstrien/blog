# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0",
#   "huggingface-hub>=0.25",
#   "pillow>=10.4",
# ]
# ///
"""Upload a sample of NLS `new-index-cards/` images as a PRIVATE training pool to silver-label.

Uses the `new_XXXX.jpg` set (distinct from the 0XXX eval cards → no leakage), uniform 8% crop +
≤1024 resize to match the eval-input preprocessing. Output: davanstrien/nls-cards-train-pool.

Usage: uv run prepare_nls_train_pool.py --repo davanstrien/nls-cards-train-pool --n 400
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

POOL = Path.home() / "Documents/nls-work/nls-metadata-extraction/data/new-index-cards"


def crop_uniform(img, margin_pct: float = 0.08):
    w, h = img.size
    mx, my = int(w * margin_pct), int(h * margin_pct)
    return img.crop((mx, my, w - mx, h - my))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-size", type=int, default=1024)
    args = ap.parse_args()

    files = sorted(POOL.glob("new_*.jpg"))[: args.n]
    rows = []
    for p in files:
        try:
            img = PILImage.open(p).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        img = crop_uniform(img)
        img.thumbnail((args.max_size, args.max_size), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        rows.append({"filename": p.name, "image": {"bytes": buf.getvalue(), "path": p.name}})
    print(f"prepared {len(rows)} training-pool images")

    features = Features({"filename": Value("string"), "image": Image()})
    Dataset.from_list(rows, features=features).push_to_hub(args.repo, private=True)
    print(f"published {len(rows)} -> {args.repo} (private)")


if __name__ == "__main__":
    main()
