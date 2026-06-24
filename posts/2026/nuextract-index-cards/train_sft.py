#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "huggingface_hub",
#     "trackio[gpu]",
#     "wandb",
#     "flash-linear-attention",
#     "causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0%2Bcu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
# ]
#
# [tool.uv]
# override-dependencies = ["transformers==5.2.0", "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0"]
# ///
"""SFT for NuExtract-3 (Qwen3.5-4B VLM) on Teklia French index cards.

Forked from iconclass-qwen35/train_sft.py. Differences:
  - base model = numind/NuExtract3 (already has a chat template — no base-clone needed).
  - dataset is the *simple* prepared train set {image, target, gt_xml}; messages are built
    on the fly with the SHARED prompt (kie_score.build_user_text), so SFT/GRPO/eval match.
  - kie_score + flat_schema.json come from the mounted bucket (/mnt/code), single source.

Usage (smoke):
    hf jobs uv run --flavor a100-large -s HF_TOKEN -v hf://buckets/davanstrien/nuextract-cards:/mnt \
        --timeout 2h -d train_sft.py -- \
        --dataset davanstrien/teklia-nuextract3-flat-train --num-samples 5 --max-steps 2 \
        --output-repo davanstrien/nuextract3-teklia-sft-smoke --report-to none --no-push
"""

import sys

sys.stdout.reconfigure(line_buffering=True)

import unsloth  # noqa: F401, E402 — must be early

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

from huggingface_hub import login  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="SFT NuExtract-3 on Teklia index cards")
    p.add_argument("--base-model", default="numind/NuExtract3")
    p.add_argument("--dataset", default="davanstrien/cards-generalist-train")
    p.add_argument("--schema", default="/mnt/code/flat_schema.json")
    p.add_argument("--code-dir", default="/mnt/code", help="dir holding kie_score.py + schema")
    p.add_argument("--output-repo", required=True)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None, help="override epochs (smoke)")
    p.add_argument("--eval-split", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--load-in-4bit", action="store_true", default=False)
    p.add_argument("--report-to", default="trackio", choices=["trackio", "wandb", "none"])
    p.add_argument("--trackio-space", default="davanstrien/trackio")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-push", action="store_true", help="skip pushing the merged model (smoke)")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    sys.path.insert(0, args.code_dir)
    import kie_score as ks

    from unsloth import FastVisionModel, UnslothVisionDataCollator
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    schema = json.load(open(args.schema, encoding="utf-8"))
    user_text = ks.build_user_text(schema)
    print(f"[1/5] Loading model: {args.base_model}  (schema {len(schema)} keys)")

    model, processor = FastVisionModel.from_pretrained(
        args.base_model, load_in_4bit=args.load_in_4bit
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        random_state=42,
    )

    print(f"[2/5] Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    print(f"  size: {len(dataset)}")

    def to_messages(ex):
        # multi-collection: each example carries its own schema-conditional prompt;
        # fall back to the single-schema user_text for single-collection runs / smokes.
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex.get("prompt") or user_text},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]},
            ],
            "images": [ex["image"]],
        }

    keep_cols = dataset.column_names
    dataset = dataset.map(to_messages, remove_columns=keep_cols)

    eval_dataset = None
    train_dataset = dataset
    if args.eval_split > 0 and len(dataset) > 10:
        sp = dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_dataset, eval_dataset = sp["train"], sp["test"]
    print(f"  train: {len(train_dataset)}  eval: {len(eval_dataset) if eval_dataset else 0}")

    report_to = [args.report_to]
    if args.report_to == "trackio":
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space
        os.environ["TRACKIO_PROJECT"] = "nuextract-teklia-sft"
        print(f"  Trackio: https://huggingface.co/spaces/{args.trackio_space}")
    elif args.report_to == "none":
        report_to = ["none"]

    output_dir = args.output_repo.split("/")[-1]
    FastVisionModel.for_training(model)

    cfg_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=42,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=report_to,
        run_name=args.run_name or "nuextract-teklia-sft",
        push_to_hub=False,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=200 if eval_dataset is not None else None,
    )
    if args.max_steps is not None:
        cfg_kwargs["max_steps"] = args.max_steps
    config = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(
            model, processor,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
    )

    print("[3/5] Training...")
    start = time.time()
    result = trainer.train()
    print(f"[4/5] Done in {(time.time() - start) / 60:.1f} min  "
          f"final loss={result.metrics.get('train_loss')}")

    if args.no_push:
        print("[5/5] --no-push set; skipping model push (smoke).")
        return
    print(f"[5/5] Saving + pushing merged model -> {args.output_repo}")
    model.save_pretrained_merged(output_dir, processor)
    model.push_to_hub_merged(args.output_repo, tokenizer=processor)
    processor.push_to_hub(args.output_repo)
    print(f"Done! https://huggingface.co/{args.output_repo}")


if __name__ == "__main__":
    main()
