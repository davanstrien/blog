# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "toolz",
#     "torch",
#     "numind",
# ]
# ///
"""Self-contained zero-shot NuExtract-3 baseline on the Teklia index cards.

Runs NuExtract-3 (vLLM) over the held-out eval cards with the flat-by-role template,
scores **in-job** with the canonical kie_score (imported from the mounted bucket), and
writes results **durably** so a number is waiting even if no one is online:
  - prints HEADLINE F1/P/R + parse-error rate to the job log (primary, always works)
  - writes baseline_metrics.json / predictions.jsonl / examples.json to the mounted bucket

Designed to run on HF Jobs with the vLLM image and the project bucket mounted, e.g.:

    hf jobs uv run --image vllm/vllm-openai:latest --flavor a100-large \
        --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
        -s HF_TOKEN -v hf://buckets/davanstrien/nuextract-cards:/mnt --timeout 1h -d \
        eval_zeroshot.py -- \
        --eval-dataset davanstrien/teklia-nuextract3-eval-input --split train \
        --schema /mnt/code/flat_schema.json --key-map /mnt/code/key_map.json \
        --out-dir /mnt/runs/baseline-nothink

Model: numind/NuExtract3 (Apache-2.0).
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
from collections import defaultdict
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from toolz import partition_all

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams  # noqa: E402

MODEL_DEFAULT = "numind/NuExtract3"


def image_to_data_uri(image) -> str:
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, dict) and "bytes" in image:
        pil = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_message(image, user_text=None):
    content = [{"type": "image_url", "image_url": {"url": image_to_data_uri(image)}}]
    if user_text:  # generic prompt-mode: schema in the message (matches how trained models see it)
        content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def split_answer(text: str) -> str:
    return text.split("</think>", 1)[-1].strip() if "</think>" in text else text.strip()


def durable_write(path: str, data: str, retries: int = 4) -> bool:
    """Write to the (possibly cold-mounted) bucket with retries. Never raises."""
    for attempt in range(1, retries + 1):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            return True
        except OSError as e:
            print(f"  write {path} attempt {attempt}/{retries} failed: {e}", flush=True)
            time.sleep(5 * attempt)
    return False


def main(args) -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA not available; this job needs a GPU.")
    login(os.environ.get("HF_TOKEN"))

    sys.path.insert(0, os.path.dirname(args.key_map))  # import canonical scorer from bucket
    import kie_score as ks

    schema = json.load(open(args.schema, encoding="utf-8"))
    key_map = json.load(open(args.key_map, encoding="utf-8"))
    print(f"Loaded schema ({len(schema)} keys) + key_map from the mounted bucket.", flush=True)

    ds = load_dataset(args.eval_dataset, split=args.split)
    print(f"Eval cards: {len(ds)} from {args.eval_dataset}:{args.split}", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.8, limit_mm_per_prompt={"image": 1})
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    # native: NuExtract's own template mechanism (zero-shot reference).
    # generic: schema embedded in the user message — matches how SFT/GRPO models are trained,
    #          so trained models AND a matched zero-shot baseline are scored consistently.
    ctk = {"enable_thinking": args.enable_thinking}
    user_text = None
    if args.prompt_mode == "native":
        ctk["template"] = json.dumps(schema, indent=4)
        if args.instructions:
            ctk["instructions"] = args.instructions
    else:
        user_text = ks.build_user_text(schema)
    print(f"prompt-mode={args.prompt_mode}  model={args.model}", flush=True)

    preds: list[str] = []
    for batch in partition_all(args.batch_size, range(len(ds))):
        batch = list(batch)
        msgs = [make_message(ds[i]["image"], user_text) for i in batch]
        outs = llm.chat(msgs, sampling, chat_template_kwargs=ctk,
                        chat_template_content_format="openai")
        preds.extend(split_answer(o.outputs[0].text) for o in outs)
        print(f"  {len(preds)}/{len(ds)} cards done", flush=True)

    # --- score (collection-aware) ---
    rows = []
    if args.collection == "teklia":
        agg_tp = agg_fp = agg_fn = 0
        exact_f1s, typed_f1s, parse_err = [], [], 0
        by_field = defaultdict(lambda: [0, 0, 0])
        routing_correct = n_routing = 0
        for i, ex in enumerate(ds):
            gt_triples = ks.xml_gt_to_triples(ex["gt_xml"])
            pred_obj = ks.parse_pred(preds[i])
            if not pred_obj:
                parse_err += 1
            pred_triples = ks.flat_pred_to_triples(pred_obj, key_map)
            ex_s = ks.score(gt_triples, pred_triples, "exact")
            ty_s = ks.score(gt_triples, pred_triples, "typed")
            agg_tp += ex_s["tp"]
            agg_fp += ex_s["fp"]
            agg_fn += ex_s["fn"]
            exact_f1s.append(ex_s["f1"])
            typed_f1s.append(ty_s["f1"])
            for t in gt_triples & pred_triples:
                by_field[t[1]][0] += 1
            for t in pred_triples - gt_triples:
                by_field[t[1]][1] += 1
            for t in gt_triples - pred_triples:
                by_field[t[1]][2] += 1
            gt_rt = next((v for (idk, f, v) in gt_triples if idk == () and f == "type_acte"), "")
            pred_rt = ks._norm(pred_obj.get("type_acte", "")) if isinstance(pred_obj, dict) else ""
            if pred_rt:
                n_routing += 1
                routing_correct += int(pred_rt == gt_rt)
            rows.append({"id": ex.get("record_id"), "score": ty_s["f1"],
                         "gt": ex["gt_xml"], "prediction": preds[i]})
        P, R, F = ks.prf(agg_tp, agg_fp, agg_fn)
        metrics = {
            "exact_micro": {"precision": P, "recall": R, "f1": F},
            "exact_macro_f1": statistics.mean(exact_f1s), "typed_macro_f1": statistics.mean(typed_f1s),
            "routing_accuracy": routing_correct / n_routing if n_routing else 0.0,
            "parse_error_rate": parse_err / len(ds),
            "by_field_f1": {f: ks.prf(*v)[2] for f, v in sorted(by_field.items())},
        }
        headline = ("exact micro-F1=%.4f  typed macro-F1=%.4f  routing=%.4f  parse-err=%.4f"
                    % (F, metrics["typed_macro_f1"], metrics["routing_accuracy"], metrics["parse_error_rate"]))
    elif args.collection == "flat":
        ftypes = ks.schema_field_types(schema)
        f1s, precs, recs = [], [], []
        for i, ex in enumerate(ds):
            sc = ks.score_flat(json.loads(ex["gt_json"]), preds[i], ftypes)
            f1s.append(sc["f1"]); precs.append(sc["precision"]); recs.append(sc["recall"])
            rows.append({"id": ex.get("image_id"), "score": sc["f1"],
                         "gt": ex["gt_json"], "prediction": preds[i]})
        parse_err = sum(1 for p in preds if not ks.parse_pred(p)) / len(ds) if len(ds) else 0.0
        metrics = {"macro_f1": statistics.mean(f1s) if f1s else 0.0,
                   "macro_precision": statistics.mean(precs) if precs else 0.0,
                   "macro_recall": statistics.mean(recs) if recs else 0.0,
                   "parse_error_rate": parse_err}
        headline = "macro-F1=%.4f  P=%.4f  R=%.4f  parse-err=%.4f" % (
            metrics["macro_f1"], metrics["macro_precision"], metrics["macro_recall"], parse_err)
    else:  # nls
        keys = ["json_extracted", "heading_fuzzy", "heading_type_exact", "epithet_fuzzy",
                "has_corrections_exact", "ms_no_f1", "ms_no_recall", "ms_no_precision",
                "folios_f1", "description_fuzzy", "entry_count_exact", "accuracy", "retrieval_score"]
        per = []
        for i, ex in enumerate(ds):
            sc = ks.score_nls(json.loads(ex["gt_json"]), preds[i])
            per.append(sc)
            rows.append({"id": ex.get("image_id"), "score": sc["retrieval_score"],
                         "gt": ex["gt_json"], "prediction": preds[i]})
        means = {k: statistics.mean([c[k] for c in per]) for k in keys}
        metrics = {**means, "parse_error_rate": 1.0 - means["json_extracted"]}
        headline = ("retrieval=%.4f  ms_no_f1=%.4f  heading_fuzzy=%.4f  folios_f1=%.4f  acc=%.4f"
                    % (means["retrieval_score"], means["ms_no_f1"], means["heading_fuzzy"],
                       means["folios_f1"], means["accuracy"]))

    metrics = {"model": args.model, "collection": args.collection, "prompt_mode": args.prompt_mode,
               "thinking": args.enable_thinking, "n_cards": len(ds),
               "timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
    print("\n" + "=" * 60, flush=True)
    print("HEADLINE  " + headline, flush=True)
    print("=" * 60 + "\n", flush=True)

    # --- durable artifacts to the bucket ---
    ok = durable_write(os.path.join(args.out_dir, "metrics.json"),
                       json.dumps(metrics, ensure_ascii=False, indent=2))
    durable_write(os.path.join(args.out_dir, "predictions.jsonl"),
                  "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    order = sorted(range(len(rows)), key=lambda i: rows[i]["score"])
    picks = order[:3] + order[-3:]
    durable_write(os.path.join(args.out_dir, "examples.json"),
                  json.dumps([rows[i] for i in picks], ensure_ascii=False, indent=2))
    print(f"Artifacts -> {args.out_dir}: {'OK' if ok else 'FAILED (see log)'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dataset", default="davanstrien/teklia-nuextract3-eval-input")
    ap.add_argument("--split", default="train")
    ap.add_argument("--schema", default="/mnt/code/flat_schema.json")
    ap.add_argument("--key-map", default="/mnt/code/key_map.json")
    ap.add_argument("--out-dir", default="/mnt/runs/baseline-nothink")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--collection", default="teklia", choices=["teklia", "nls", "flat"],
                    help="scoring: teklia (gt_xml triples), nls (score_nls), flat (any flat gt_json schema)")
    ap.add_argument("--prompt-mode", default="native", choices=["native", "generic"],
                    help="native=NuExtract template kwarg (zero-shot ref); "
                         "generic=schema in user message (matches trained models)")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--instructions", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=16)
    main(ap.parse_args())
