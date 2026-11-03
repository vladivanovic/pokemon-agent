#!/usr/bin/env python3
"""Fine-tune the Pokemon supervisor: LoRA SFT on Qwen3.5-4B.

Runs on the ORIN in hermes_venv (transformers 5.17 / TRL 1.15 /
PEFT 0.21.2 / torch 2.14.1+cu132) - the dev VM has no GPU, so this
script is written for that stack and only argument-parsed here.

Data: the chat-messages JSONL built by
scripts/build_supervisor_dataset.py (dataset/supervisor/train.jsonl +
eval.jsonl). The examples are short - a compact state summary in, one
goal JSON out - so max_len 4096 is generous and no length ladder is
needed (unlike the Metasploit run, whose trajectories are the point).

    # one-time: fetch the base weights
    python3 scripts/train_supervisor_lora.py --download

    # train (defaults: LoRA r=32, bf16, lr 1e-4, 2 epochs)
    python3 scripts/train_supervisor_lora.py

    # where things live somewhere else
    python3 scripts/train_supervisor_lora.py --data-dir /path/to/dataset/supervisor

See docs/supervisor-training.md for the full pipeline and eval plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_BASE = "Qwen/Qwen3.5-4B"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="LoRA SFT for the Pokemon supervisor (Qwen3.5-4B).")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="HF repo id or local path of the base model "
                         f"(default {DEFAULT_BASE})")
    ap.add_argument("--model-dir", default="models/Qwen3.5-4B",
                    help="where --download puts the base weights, and "
                         "the first place training looks for them "
                         "(default models/Qwen3.5-4B)")
    ap.add_argument("--download", action="store_true",
                    help="snapshot-download the base weights to "
                         "--model-dir before training")
    ap.add_argument("--data-dir", default="dataset/supervisor",
                    help="dir with train.jsonl / eval.jsonl "
                         "(default dataset/supervisor)")
    ap.add_argument("--out", default="training_output/qwen_supervisor",
                    help="checkpoint output dir "
                         "(default training_output/qwen_supervisor)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=1,
                    help="per-device train batch size (default 1)")
    ap.add_argument("--ga", type=int, default=4,
                    help="gradient accumulation steps (default 4)")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--completion-only", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="train on assistant tokens only (TRL "
                         "assistant_only_loss; needs the chat template's "
                         "generation markers - Qwen's has them. Use "
                         "--no-completion-only if a future template "
                         "drops them)")
    ap.add_argument("--liger", action="store_true",
                    help="use the Liger kernel (fused CE etc.), as in "
                         "the Metasploit runs; off by default so the "
                         "script also runs where liger-kernel is absent")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Heavy imports live here so --help works on GPU-less machines.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_dir = Path(args.model_dir)
    if args.download:
        from huggingface_hub import snapshot_download
        print(f"downloading {args.base} -> {model_dir}")
        snapshot_download(repo_id=args.base, local_dir=str(model_dir))
    model_path = str(model_dir) if (model_dir / "config.json").exists() \
        else args.base
    print(f"base weights: {model_path}")

    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.jsonl"
    if not train_file.exists():
        print(f"missing {train_file} - build the dataset first:\n"
              "  python3 scripts/build_supervisor_dataset.py",
              file=sys.stderr)
        return 2
    files = {"train": str(train_file)}
    eval_file = data_dir / "eval.jsonl"
    if eval_file.exists():
        files["eval"] = str(eval_file)
    ds = load_dataset("json", data_files=files)
    print(f"dataset: { {k: len(v) for k, v in ds.items()} }")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        attn_implementation=args.attn)
    model.config.use_cache = False  # with gradient checkpointing

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    config = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch" if "eval" in ds else "no",
        bf16=True,
        max_length=args.max_len,
        packing=False,
        assistant_only_loss=args.completion_only,
        use_liger_kernel=args.liger,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    final_dir = str(Path(args.out) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"adapter saved to {final_dir}")
    print("next: evaluate on the held-out split + stuck-state battery "
          "(docs/supervisor-training.md), then serve behind LLMPlanner "
          "with the validator still in front.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
