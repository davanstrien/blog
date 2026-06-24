# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "toolz",
#     "torch",
# ]
# ///
"""Qwen3-VL-8B (vLLM) on index cards — silver-label a training pool, or score a baseline.

Two modes:
  --mode label : run the NLS extraction prompt over an image pool, keep index_card extractions,
                 push a PRIVATE dataset {filename, image, extraction} for SFT (NLS silver labels).
  --mode score : score predictions against GT (score_nls) and write metrics to the bucket
                 (gives the Qwen3-VL-8B baseline on our eval-input / a Teklia cross-number).

The labeler is calibrated: Qwen3-VL-8B scores ~0.76 retrieval on the 103-card NLS GT.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from toolz import partition_all

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams  # noqa: E402

# NLS Advocates extraction prompt — vendored from nls vlm_eval_v2.py (finalised).
NLS_PROMPT = """Analyze this image from a historical library index card collection.

IMPORTANT: These cards are physically stacked. You may see FAINT or LIGHTER text bleeding through
from cards behind this one. ONLY extract from the PRIMARY card's CLEAR, DARK, SHARP text.

FIRST classify image_type: "index_card" (clear heading in caps + a manuscript number, reads
left-to-right), "verso" (mirrored/blank back), "cover", "blank", or "other".

If index_card, extract:
1. heading: the PRIMARY heading only, stop BEFORE any underlined epithet. Preserve abbreviations
   and punctuation exactly ("Abbate, A." not "ABBATE (Antonie)"). Do NOT expand initials.
2. heading_type: one of person | family | corporate | geographic | subject.
3. epithet: ALL underlined text (title, occupation, role, qualifier).
4. has_corrections: true ONLY for handwritten text modifying/correcting typed text.
5. indicates_continuation: true if any "cont/", "continued", or continuation marker.
6. entries: array of manuscript references, each with:
   - ms_no: manuscript number incl. any prefix exactly ("5538", "Ch. 8629"); no trailing period.
   - folios: page references ONLY as an array; almost always start with f./ff./p./pp./no./nos.;
     [] if none; never put descriptions or ms numbers here.
   - description: document description with date ("letter of (1783)").

For non-index_card types set the extraction fields to null/empty.
Respond with ONLY valid JSON: {"image_type","heading","heading_type","epithet","has_corrections",
"indicates_continuation","entries":[{"ms_no","folios","description"}],"notes"}."""


def to_data_uri(image) -> str:
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"]))
    pil = image.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_msg(image, prompt):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": to_data_uri(image)}},
        {"type": "text", "text": prompt}]}]


def parse_json(text: str):
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:  # noqa: BLE001
        return None


def main(args) -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    login(os.environ.get("HF_TOKEN"))
    sys.path.insert(0, args.code_dir)
    import kie_score as ks  # noqa: F401  (used in score mode)

    prompt = NLS_PROMPT if args.collection == "nls" else ks.build_user_text(
        json.load(open(args.schema, encoding="utf-8")))
    ds = load_dataset(args.dataset, split=args.split)
    print(f"{len(ds)} images from {args.dataset}; mode={args.mode}", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1})
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    preds = []
    for batch in partition_all(args.batch_size, range(len(ds))):
        batch = list(batch)
        outs = llm.chat([make_msg(ds[i][args.image_column], prompt) for i in batch], sp,
                        chat_template_content_format="openai")
        preds.extend(o.outputs[0].text for o in outs)
        print(f"  {len(preds)}/{len(ds)}", flush=True)

    if args.mode == "label":
        objs = [parse_json(p) for p in preds]
        skip_types = {"blank", "other", "verso", "divider", "index_divider", "see_reference"}

        def has_content(o):
            if not isinstance(o, dict) or not o:
                return False
            ct = str(o.get("card_type") or o.get("image_type") or "").strip().lower()
            if ct in skip_types:
                return False
            scalars = [v for k, v in o.items() if k not in ("card_type", "image_type")
                       and not isinstance(v, (list, dict)) and str(v or "").strip()]
            return len(scalars) >= 2  # at least 2 real fields → a usable record

        keep = [i for i, o in enumerate(objs) if has_content(o)]
        print(f"kept {len(keep)}/{len(ds)} content extractions", flush=True)
        sub = ds.select(keep)
        sub = sub.add_column("extraction", [json.dumps(objs[i], ensure_ascii=False) for i in keep])
        sub.push_to_hub(args.out_repo, private=True)
        print(f"pushed {len(sub)} -> {args.out_repo} (private)", flush=True)
    else:  # score
        keys = ["retrieval_score", "ms_no_f1", "heading_fuzzy", "heading_type_exact",
                "epithet_fuzzy", "folios_f1", "accuracy"]
        per = [ks.score_nls(json.loads(ds[i]["gt_json"]), preds[i]) for i in range(len(ds))]
        m = {k: statistics.mean([p[k] for p in per]) for k in keys}
        m.update({"model": args.model, "collection": args.collection, "n": len(ds)})
        print("HEADLINE  " + "  ".join("%s=%.4f" % (k, m[k]) for k in keys), flush=True)
        os.makedirs(args.out_dir, exist_ok=True)
        json.dump(m, open(os.path.join(args.out_dir, "metrics.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--mode", choices=["label", "score"], default="label")
    ap.add_argument("--collection", default="nls", choices=["nls", "teklia"])
    ap.add_argument("--schema", default="/mnt/code/flat_schema.json")
    ap.add_argument("--code-dir", default="/mnt/code")
    ap.add_argument("--out-repo", default="davanstrien/nls-cards-silver")
    ap.add_argument("--out-dir", default="/mnt/runs/qwen-score")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    main(ap.parse_args())
