#!/usr/bin/env python3
"""LoRA SFT for the Gen 1 battle advisor.

DO NOT RUN until evaluate_headroom.py shows real headroom AND
data/battles.jsonl has >5k pairs. See README.md for the gate.

Text-only: loads Qwen3.5-0.8B as Qwen3_5ForCausalLM, which drops the vision
tower (752M params vs the full multimodal checkpoint) - battles need no pixels,
and the VL weights stay untouched for future navigation work.

Usage:
    python3 train_sft.py --data data/battles_chat.jsonl --out out/qwen08-battle-lora
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, Qwen3_5ForCausalLM,
                          TrainingArguments)
from trl import SFTTrainer

MODEL = "/home/vlad/models/Qwen3.5-0.8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default="data/battles_chat.jsonl")
    ap.add_argument("--out", default="out/qwen08-battle-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.data, split="train")

    def to_text(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        processing_class=tok,
        max_seq_length=args.max_len,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
