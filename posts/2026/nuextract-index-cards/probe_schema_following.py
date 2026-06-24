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
"""Schema-following probe on an UNSEEN collection (no GT needed) — the forgetting test.

Measures the RL-learned behavior NuExtract-3 ships with: given a novel schema, does the model
emit valid JSON whose keys conform to the schema, with sensible coverage and clean termination?
Run on base vs fine-tuned with the same schema → the delta is the forgetting (or transfer).

Metrics: parse_rate, key_conformance (pred keys ⊆ schema), mean_field_coverage (non-empty
schema keys / schema size), schema_keys_hallucinated. Predictions dumped for later silver+
scoring. Works in generic or native prompt mode.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from toolz import partition_all

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams  # noqa: E402


def to_uri(image) -> str:
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"]))
    pil = image.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def durable_write(path: str, data: str, retries: int = 4) -> bool:
    for attempt in range(1, retries + 1):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            return True
        except OSError as e:
            print(f"  write retry {attempt}: {e}", flush=True)
            time.sleep(5 * attempt)
    return False


def main(args) -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    login(os.environ.get("HF_TOKEN"))
    sys.path.insert(0, args.code_dir)
    import kie_score as ks

    schema = json.load(open(args.schema, encoding="utf-8"))
    schema_keys = set(schema)
    ds = load_dataset(args.dataset, split=args.split)
    print(f"{len(ds)} cards · model={args.model} · prompt-mode={args.prompt_mode}", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=16384,
              gpu_memory_utilization=0.8, limit_mm_per_prompt={"image": 1})
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    ctk = {"enable_thinking": False}
    user_text = None
    if args.prompt_mode == "native":
        ctk["template"] = json.dumps(schema, indent=4)
    else:
        user_text = ks.build_user_text(schema)

    def msg(image):
        content = [{"type": "image_url", "image_url": {"url": to_uri(image)}}]
        if user_text:
            content.append({"type": "text", "text": user_text})
        return [{"role": "user", "content": content}]

    preds = []
    for batch in partition_all(args.batch_size, range(len(ds))):
        batch = list(batch)
        outs = llm.chat([msg(ds[i]["image"]) for i in batch], sp,
                        chat_template_kwargs=ctk, chat_template_content_format="openai")
        preds.extend(o.outputs[0].text.split("</think>")[-1].strip() for o in outs)
        print(f"  {len(preds)}/{len(ds)}", flush=True)

    parse_ok, conf, cov, halluc = [], [], [], []
    rows = []
    for i, p in enumerate(preds):
        obj = ks.parse_pred(p)
        ok = isinstance(obj, dict) and bool(obj)
        parse_ok.append(float(ok))
        if ok:
            keys = list(obj.keys())
            in_schema = [k for k in keys if k in schema_keys or ks._key_norm(k) in
                         {ks._key_norm(s) for s in schema_keys}]
            conf.append(len(in_schema) / len(keys) if keys else 1.0)
            halluc.append(len(keys) - len(in_schema))
            nonempty = sum(1 for k in in_schema if str(obj.get(k) or "").strip()
                           not in ("", "[]", "None", "null"))
            cov.append(nonempty / len(schema_keys))
        rows.append({"image_id": ds[i].get("image_id"), "prediction": p})

    metrics = {
        "model": args.model, "dataset": args.dataset, "prompt_mode": args.prompt_mode,
        "n": len(ds), "schema_keys": len(schema_keys),
        "parse_rate": statistics.mean(parse_ok),
        "key_conformance": statistics.mean(conf) if conf else 0.0,
        "mean_field_coverage": statistics.mean(cov) if cov else 0.0,
        "mean_hallucinated_keys": statistics.mean(halluc) if halluc else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print("\n" + "=" * 60, flush=True)
    print("HEADLINE  parse=%.3f  conformance=%.3f  coverage=%.3f  halluc-keys=%.2f"
          % (metrics["parse_rate"], metrics["key_conformance"],
             metrics["mean_field_coverage"], metrics["mean_hallucinated_keys"]), flush=True)
    print("=" * 60, flush=True)
    durable_write(os.path.join(args.out_dir, "metrics.json"),
                  json.dumps(metrics, ensure_ascii=False, indent=2))
    durable_write(os.path.join(args.out_dir, "predictions.jsonl"),
                  "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"Artifacts -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="davanstrien/rubenstein-probe-input")
    ap.add_argument("--split", default="train")
    ap.add_argument("--schema", default="/mnt/code/rubenstein_schema.json")
    ap.add_argument("--code-dir", default="/mnt/code")
    ap.add_argument("--prompt-mode", default="generic", choices=["generic", "native"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    main(ap.parse_args())
