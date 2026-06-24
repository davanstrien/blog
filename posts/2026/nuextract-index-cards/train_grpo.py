#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth<2026.6",
#     "unsloth-zoo<2026.6",
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
"""GRPO for NuExtract-3 (Qwen3.5-4B VLM) on Teklia index cards with the typed KIE reward.

Forked from iconclass-qwen35/train_grpo.py. Keeps the load/shim/GRPOConfig/save scaffold;
**replaces** the iconclass rewards with kie_score's typed partial-credit field-F1 (the eval
metric, run online). The reward is the load-bearing signal; format/schema are optional gates.

Reward arms (`--rewards`, default `typed_f1`):
  - typed_f1   — typed partial credit, free-text = 0.85*(1-CER)+0.15*exact  [PRIMARY]
  - typed_anls — free-text = ANLS (0.5 cliff)            [ablation]
  - exact_f1   — exact match on every field              [ablation, Medical-VIE/ThinkJSON style]
  - format     — valid-JSON gate (saturates -> dead weight; off by default)
  - schema     — fraction of keys in schema (gate)

Usage (smoke):
    hf jobs uv run --flavor a100-large -s HF_TOKEN -v hf://buckets/davanstrien/nuextract-cards:/mnt \
        --timeout 2h -d train_grpo.py -- \
        --model numind/NuExtract3 --dataset davanstrien/teklia-nuextract3-flat-train \
        --num-samples 8 --num-generations 4 --max-steps 5 \
        --output-repo davanstrien/nuextract3-teklia-grpo-smoke --no-push
"""

import unsloth  # noqa: F401 — must be first, patches transformers

import argparse
import json
import os
import sys

from datasets import load_dataset
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from unsloth import FastVisionModel

# --- Compat shim: unsloth forwards `mm_token_type_ids` into generate(); transformers 5.2.0
# rejects it though the model ignores it. Drop it from the validation check. ---
import transformers.generation.utils as _gen_utils  # noqa: E402

_orig_validate_model_kwargs = _gen_utils.GenerationMixin._validate_model_kwargs


def _validate_model_kwargs_drop_mmtti(self, model_kwargs, *args, **kwargs):
    if isinstance(model_kwargs, dict):
        model_kwargs.pop("mm_token_type_ids", None)
    return _orig_validate_model_kwargs(self, model_kwargs, *args, **kwargs)


_gen_utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_drop_mmtti


class RewardHealthCallback(TrainerCallback):
    """Fire Trackio alerts on the GRPO failure modes (dead reward variance / flat reward)."""

    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rstd = logs.get("reward_std")
        rmean = logs.get("reward")
        try:
            import trackio
            if rstd is not None and rstd < 1e-4 and state.global_step > 2:
                trackio.alert(title="reward_std~0",
                              text=f"step {state.global_step}: reward_std={rstd:.2e} — no contrastive signal",
                              level="WARN")
            if rmean is not None:
                self.history.append(rmean)
                if len(self.history) >= 8 and max(self.history[-8:]) - min(self.history[-8:]) < 1e-3:
                    trackio.alert(title="reward flat",
                                  text=f"step {state.global_step}: reward flat over 8 logs (~{rmean:.3f})",
                                  level="WARN")
        except Exception as e:  # noqa: BLE001
            print(f"[alert] {e}")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description="GRPO NuExtract-3 on Teklia (typed KIE reward)")
    p.add_argument("--model", default="numind/NuExtract3", help="start model (or an SFT checkpoint)")
    p.add_argument("--dataset", default="davanstrien/teklia-nuextract3-flat-train")
    p.add_argument("--schema", default="/mnt/code/flat_schema.json")
    p.add_argument("--key-map", default="/mnt/code/key_map.json")
    p.add_argument("--code-dir", default="/mnt/code")
    p.add_argument("--output-repo", required=True)
    p.add_argument("--rewards", nargs="+", default=["typed_f1"])
    p.add_argument("--prompt-mode", default="generic", choices=["generic", "native"],
                   help="native = pre-rendered NuExtract structured-mode prompt (the GRPO re-test)")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--report-to", default="trackio", choices=["trackio", "none"])
    p.add_argument("--trackio-space", default="davanstrien/trackio")
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    sys.path.insert(0, args.code_dir)
    import kie_score as ks

    schema = json.load(open(args.schema, encoding="utf-8"))
    ks.set_key_map(json.load(open(args.key_map, encoding="utf-8")))
    user_text = ks.build_user_text(schema)
    print(f"[1/4] Model: {args.model}  schema {len(schema)} keys  rewards {args.rewards}")

    model, processor = FastVisionModel.from_pretrained(args.model, load_in_4bit=False)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,          # language-layers-only per the GRPO recipe
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0, bias="none", random_state=42,
    )

    # --- rewards (kie_score is the single source) ---
    def typed_f1(completions, ground_truth=None, **kw):
        return ks.typed_field_f1_reward(completions, ground_truth=ground_truth, string_metric="cer")

    def typed_anls(completions, ground_truth=None, **kw):
        return ks.typed_field_f1_reward(completions, ground_truth=ground_truth, string_metric="anls")

    def exact_f1(completions, ground_truth=None, **kw):
        return ks.typed_field_f1_reward(completions, ground_truth=ground_truth, string_metric="exact")

    reward_map = {
        "typed_f1": typed_f1, "typed_anls": typed_anls, "exact_f1": exact_f1,
        "format": ks.format_reward, "schema": ks.schema_conformance_reward,
    }
    reward_funcs = [reward_map[r] for r in args.rewards]
    print(f"[2/4] Reward fns: {[r for r in args.rewards]}")

    # --- dataset: {prompt, image, ground_truth=gt_xml} ---
    train_ds = load_dataset(args.dataset, split="train")
    if args.num_samples is not None:
        train_ds = train_ds.select(range(min(args.num_samples, len(train_ds))))

    # native mode: pre-render NuExtract's own structured-mode prompt (【template_start】… markers)
    # as a raw string with vision tokens baked in — GRPOTrainer passes raw-string prompts through
    # untouched, so the model sees its true in-distribution format (the collator can't forward
    # template= kwargs itself). generic mode keeps the schema-in-message framing.
    native_prompt = None
    if args.prompt_mode == "native":
        native_prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}]}],
            template=json.dumps(schema, indent=4),
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=False,
        )
        print(f"[native prompt head] {native_prompt[:200]!r}")

    def transform(ex):
        if native_prompt is not None:
            prompt = native_prompt
        else:
            prompt = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": user_text}]}]
        return {
            "prompt": prompt,
            "image": ex["image"],
            "ground_truth": ex["gt_xml"],
        }

    drop = [c for c in train_ds.column_names if c not in ("image",)]
    dataset = train_ds.map(transform, remove_columns=drop)
    print(f"[3/4] Train prompts: {len(dataset)}")

    report_to = ["trackio"] if args.report_to == "trackio" else ["none"]
    if args.report_to == "trackio":
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space
        os.environ["TRACKIO_PROJECT"] = "nuextract-teklia-grpo"

    grpo_kwargs = dict(
        output_dir=args.output_repo.split("/")[-1],
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        remove_unused_columns=False,
        report_to=report_to,
        logging_steps=1,
        save_strategy="steps",
        save_steps=200,
        log_completions=True,
        num_completions_to_print=2,
        push_to_hub=not args.no_push,
        hub_model_id=args.output_repo,
        hub_strategy="checkpoint",
        gradient_checkpointing=True,
    )
    if args.max_steps is not None:
        grpo_kwargs["max_steps"] = args.max_steps
        grpo_kwargs["num_train_epochs"] = 100
    training_args = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        callbacks=[RewardHealthCallback()],
    )

    print("[4/4] GRPO training...")
    trainer.train()

    if args.no_push:
        print("--no-push set; skipping merged push (smoke).")
        return
    print(f"Saving merged model -> {args.output_repo}")
    model.save_pretrained_merged(args.output_repo.split("/")[-1] + "-merged", processor)
    model.push_to_hub_merged(args.output_repo, tokenizer=processor)
    print("Done!")


if __name__ == "__main__":
    main()
