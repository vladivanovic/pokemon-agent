#!/usr/bin/env python3
"""Train Qwen2.5-VL-7B LoRA on TAS replay data (frame -> button action).

Dataset layout (produced by tas_replay.py):
    dataset/tas_blue/
        frames/frame_000000.jpg ...
        manifest.jsonl  {"frame": N, "file": "frame_XXXXXX.jpg", "buttons": ["up", ...]}

Task framing for the VLM:
    prompt:  the current frame + a compact instruction
    target:  the button(s) pressed on the NEXT input frame(s)

We compress the 8 button names into a compact action string, e.g. "up"
or "a+start", so the model emits a short token sequence. At inference,
the action string is mapped back to PyBoy button presses.

LoRA on Qwen2.5-VL-7B-Instruct (vision tower frozen by default in peft
target_modules selection below - we target language + merger projections).

Usage:
    python train_tas_lora.py \
        --data dataset/tas_blue \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --out out/tas_lora \
        --epochs 1 --bs 4 --accum 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ACTION_KEYS = ["up", "down", "left", "right", "start", "select", "b", "a"]
ACTION_TO_TOKEN = {
    "up": "<|up|>",
    "down": "<|down|>",
    "left": "<|left|>",
    "right": "<|right|>",
    "start": "<|start|>",
    "select": "<|select|>",
    "b": "<|b|>",
    "a": "<|a|>",
}


def buttons_to_str(buttons: List[str]) -> str:
    """['up'] -> 'up'; ['a','start'] -> 'a+start' (sorted for canonical form)."""
    present = sorted(set(buttons) & set(ACTION_KEYS))
    return "+".join(present) if present else "none"


class TasFrameDataset(Dataset):
    def __init__(self, data_dir: Path, processor, max_samples: int = 0):
        self.data_dir = data_dir
        self.processor = processor
        self.samples = []
        with (data_dir / "manifest.jsonl").open() as f:
            for line in f:
                self.samples.append(json.loads(line))
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.data_dir / "frames" / s["file"]
        action_str = buttons_to_str(s["buttons"])

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": ("What button is pressed next? "
                     "Answer with one of: up, down, left, right, start, "
                     "select, b, a, or none. Combine with + if multiple.")},
                ],
            },
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": action_str}]},
        ]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False)

        # Process the FULL conversation (prompt + answer) for input_ids
        full = self.processor(text=[full_text], images=[str(img_path)],
                              return_tensors="pt", padding=True)
        full = {k: v.squeeze(0) for k, v in full.items()}

        # Process the PROMPT ONLY to find how many tokens to mask. The image
        # expands to a deterministic number of vision tokens, so the prefix
        # length is stable between the two calls.
        prompt_only = self.processor(text=[prompt_text], images=[str(img_path)],
                                     return_tensors="pt", padding=True)
        prompt_len = int(prompt_only["input_ids"].shape[1])

        labels = full["input_ids"].clone()
        labels[:prompt_len] = -100          # mask the prompt
        labels[full["attention_mask"] == 0] = -100  # mask padding
        full["labels"] = labels
        return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/tas_blue")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", default="out/tas_lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=2, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--freeze-vision", action="store_true", default=True)
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not (data_dir / "manifest.jsonl").exists():
        raise SystemExit(f"No manifest at {data_dir}/manifest.jsonl - run tas_replay.py first")

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # LoRA: target attention + MLP projections in the language model.
    # Skip visual tower to save memory - screen reading is already decent.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = TasFrameDataset(data_dir, processor, max_samples=args.max_samples)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,  # we return dict with images
        dataloader_num_workers=0,
        # transformers 5.x removed warmup_ratio; use explicit steps
        warmup_steps=200,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.out}/adapter")
    model.save_pretrained(str(Path(args.out) / "adapter"))
    processor.save_pretrained(str(Path(args.out) / "adapter"))
    print("Done.")


if __name__ == "__main__":
    main()