# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0",
#   "huggingface-hub>=0.25",
#   "pillow>=10.4",
# ]
# ///
"""Package the NLS Advocates index-card GT (103 cards) as a PRIVATE eval-input dataset.

Mirrors the Teklia eval-input. Reads the cataloguer-reviewed GT + local card images, applies
the same uniform 8% crop + ≤1024 resize the NLS Inspect eval uses (its YOLO-free fallback), and
pushes `image / gt_json / image_id` to a private HF dataset for our eval harness.

NB: the published 94.6% number used YOLO-detector crop; we re-run Qwen3-VL-8B on THIS same
uniform-cropped eval-input ourselves so baseline and NuExtract-3 are apples-to-apples.

Usage: uv run prepare_nls_eval.py --repo davanstrien/nls-advocates-eval-input
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

NLS = Path.home() / "Documents/nls-work/nls-metadata-extraction/data/index-card-eval"


def crop_uniform(img: PILImage.Image, margin_pct: float = 0.08) -> PILImage.Image:
    w, h = img.size
    mx, my = int(w * margin_pct), int(h * margin_pct)
    return img.crop((mx, my, w - mx, h - my))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gt", default=str(NLS / "gt.json"))
    ap.add_argument("--images", default=str(NLS / "images"))
    ap.add_argument("--max-size", type=int, default=1024)
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    cards = gt["cards"]
    images_dir = Path(args.images)

    rows = []
    for c in cards:
        if c.get("image_type") != "index_card":
            continue
        p = images_dir / f"{c['image_id']}.jpg"
        if not p.exists():
            print(f"  ! missing image {p.name}, skip")
            continue
        img = PILImage.open(p).convert("RGB")
        img = crop_uniform(img)
        img.thumbnail((args.max_size, args.max_size), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        rows.append({
            "image_id": c["image_id"],
            "image": {"bytes": buf.getvalue(), "path": f"{c['image_id']}.jpg"},
            "gt_json": json.dumps(c["extraction"], ensure_ascii=False),
            "image_type": c["image_type"],
        })
    print(f"prepared {len(rows)} index_card rows")

    features = Features({
        "image_id": Value("string"),
        "image": Image(),
        "gt_json": Value("string"),
        "image_type": Value("string"),
    })
    Dataset.from_list(rows, features=features).push_to_hub(args.repo, private=args.private)
    print(f"published {len(rows)} -> {args.repo} (private={args.private})")


if __name__ == "__main__":
    main()
