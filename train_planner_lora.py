#!/usr/bin/env python3
"""Train Qwen2.5-VL-7B LoRA as a PLANNER (screen + RAM state -> next goal).

This is the strategic half of the two-tier agent:
    VLM planner  : (screen + state) -> one high-level goal  [this model]
    Von driver   : goal -> BFS/ledge/battle execution       [autopilot.py]

Training data comes from convert_pokedreamer.py (PokeDreamer NPZ -> frames +
auto-derived goals) and optionally tas_replay.py output. NO manual labeling:
the goal is a pure function of RAM state, computed offline for every frame.

Dataset layout:
    dataset/planner/
        frames/frame_000000.jpg ...
        manifest.jsonl
            {"file": "...", "goal": "...", "state": {map,badges,x,y,party_hp,...}}

Use:
    python train_planner_lora.py \
        --data dataset/planner \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --out out/planner_lora \
        --epochs 2 --bs 2 --accum 8
"""

from __future__ import annotations

import argparse
import json
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

# The VLM may only answer with a goal drawn from this fixed vocabulary. Keeping
# it closed means the driver always receives an executable instruction and the
# model can't drift into prose. Mirrors the goal branches in autopilot.py.
GOAL_VOCAB = [
    "Exit the house to Pallet Town - walk to the south door.",
    "Walk north from Pallet Town to Viridian City.",
    "Collect Oak's Parcel from Viridian Mart, then return to Oak's Lab in Pallet Town.",
    "Buy supplies at the counter (press A, choose BUY). Get Poke Balls; Pewter Mart also sells Potions.",
    "Travel north toward Pewter City - keep heading north along the path.",
    "Heal the party at the nearest Pokemon Center.",
    "Beat the Gym Leader - attack with the most effective move; heal when HP drops below a third.",
    "Exit the Gym to the city - walk to the south door and step onto it.",
    "Leave Pewter City via its EAST exit to Route 3.",
    "Head EAST along Route 3 - cross the ledges and reach Route 4 at the far east end.",
    "Exit the Pokemon Center to the map outside - walk to the exit door and step onto it.",
    "Exit the Mart to the map outside.",
    "Traverse Mt. Moon from the south entrance to the north exit, then continue to Cerulean City.",
    "Head EAST along Route 4 toward Cerulean City.",
    "Win the battle - pick the most effective move and attack.",
    "Read the dialog and advance it with A (or B to skip).",
    "Take a starter Pokemon from the table in Oak's Lab - stand facing a Pokeball, press A, confirm YES.",
    "Exit this building to Pewter City - walk to the exit door and step onto it, then head EAST to Route 3.",
    "Exit this house to the map outside - walk to the exit door.",
    "Catch an Electric-type in Viridian Forest (Pikachu 5% / Red, Electabuzz 5% / Blue) before Misty's Gym.",
    "Catch a Grass-type in Viridian Forest (Caterpie/Metapod/Red, Oddish/Blue) for Rock/Ground coverage.",
    "Catch a Water- or Grass-type in Route 3 / Viridian Forest before Misty's Gym (Bulbasaur line if starter, or Psyduck in Route 3).",
    "Enter Mt Moon 1F from Route 4 - walk up into the cave entrance and head inside.",
    "Descend from Mt Moon 1F to Mt Moon B1F.",
    "Descend from Mt Moon 1F north half through B1F to the south exits to Route 4.",
    "Cross Mt Moon B1F toward the south exits to Route 4.",
    "Climb from Mt Moon B2F back up to Mt Moon B1F.",
    "Catch an Electric- or Grass-type in Route 4 / Route 24 before Misty (Bellsprout Route 24, Oddish Route 5).",
    "Travel to Cerulean City and challenge Misty.",
    "Catch a Fire/Psychic/Flying-type for Lt. Surge (Abra Route 24/25, Pidgey Route 1/2, Vulpix Route 3/Red).",
    "Progress toward Vermilion City and Lt. Surge.",
    "Continue toward Cerulean City for badge #2.",
    "Progress toward Pewter City and the first Gym.",
]


def state_to_text(state: dict) -> str:
    """Compact, stable serialization of the RAM state for the prompt.

    Kept as sorted key=value pairs so the model sees a consistent token order -
    an unstable serialization would make the planner chase formatting noise.
    """
    party = [f"{h}/{m}" for h, m in
             zip(state.get("party_hp", []), state.get("party_max_hp", []))
             if m > 0]
    bits = [
        f"map={state.get('map', '?')}",
        f"badges={state.get('badges', 0)}",
        f"pos=({state.get('x', 0)},{state.get('y', 0)})",
        f"facing={state.get('facing', 0)}",
        f"in_battle={bool(state.get('in_battle'))}",
        f"dialog={bool(state.get('dialog_open'))}",
        f"party_hp={','.join(party) if party else 'none'}",
    ]
    return " ".join(bits)


PROMPT_TEMPLATE = (
    "You are the strategy layer of a Pokemon Red agent. Look at the game "
    "screen and the RAM state, then state the SINGLE next high-level goal. "
    "Answer with exactly one line from the allowed goal list, optionally "
    "followed by '| ACTION: <low-level-action>' for immediate execution. "
    "Nothing else.\\n\\n"
    "RAM state: {state}\\n\\n"
    "Allowed goals:\\n{vocab}\\n\\n"
    "Next goal:"
)


class PlannerDataset(Dataset):
    def __init__(self, data_dir: Path, processor, max_samples: int = 0):
        self.data_dir = data_dir
        self.processor = processor
        self.samples: List[dict] = []
        with (data_dir / "manifest.jsonl").open() as f:
            for line in f:
                self.samples.append(json.loads(line))
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} planner samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.data_dir / "frames" / s["file"]
        goal = s["goal"]
        action = s.get("action", "")
        state_txt = state_to_text(s.get("state", {}))
        vocab_txt = "\n".join(f"- {g}" for g in GOAL_VOCAB)

        # Include action in the target for tool-use training
        target = goal
        if action:
            target = f"{goal} | ACTION: {action}"

        prompt = PROMPT_TEMPLATE.format(state=state_txt, vocab=vocab_txt)
        prompt_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": prompt},
            ]},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False)

        full = self.processor(text=[full_text], images=[str(img_path)],
                              return_tensors="pt", padding=True)
        full = {k: v.squeeze(0) for k, v in full.items()}

        prompt_only = self.processor(text=[prompt_text], images=[str(img_path)],
                                     return_tensors="pt", padding=True)
        prompt_len = int(prompt_only["input_ids"].shape[1])

        labels = full["input_ids"].clone()
        labels[:prompt_len] = -100
        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        return full


class PlannerCollator:
    """Pad variable-length planner samples into a batch.

    The default collator calls torch.stack, which fails when samples have
    different token lengths (a long goal line vs a short one). This pads
    input_ids / attention_mask / labels to the batch max and stacks the
    remaining per-sample tensors (pixel_values, image_grid_thw).
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        max_len = max(f["input_ids"].shape[0] for f in features)
        # Every 1-D tensor whose length tracks the sequence (input_ids,
        # attention_mask, labels, mm_token_type_ids, ...) must be padded to the
        # same max_len. Missing one (e.g. mm_token_type_ids) makes stack fail
        # on mixed-length batches and the cat fallback silently flattens it to
        # 1-D, which the model then indexes as a scalar ("too many indices for
        # tensor of dimension 0" deep in get_rope_index).
        batch = {k: [] for k in features[0] if v_ndim(features[0][k]) == 1
                 and features[0][k].shape[0] == features[0]["input_ids"].shape[0]}
        passthrough = {}
        for f in features:
            n = f["input_ids"].shape[0]
            pad = max_len - n
            for k, v in f.items():
                if k in batch:                      # 1-D sequence tensor -> pad
                    batch[k].append(
                        torch.cat([v, torch.full((pad,), self._pad_value(k, v),
                                                  dtype=v.dtype)]))
                else:
                    passthrough.setdefault(k, []).append(v)
        out = {k: torch.stack(v) for k, v in batch.items()}
        for k, v in passthrough.items():
            if k == "pixel_values":
                # Qwen2.5-VL expects pixel_values as a FLAT (total_visual_tokens,
                # 1176) tensor concatenated across the batch - NOT stacked into
                # (B, tokens, dim). The model slices it with image_grid_thw to
                # recover per-sample vision features; a 3D stack breaks that
                # indexing ("too many indices for tensor of dimension 0").
                out[k] = torch.cat(v, dim=0)
            elif k == "image_grid_thw":
                # (B, 3) rows of (t, h, w) - get_vision_position_ids zips over
                # the rows and unpacks each into t/h/w, so the batch dim must
                # stay 2D. Stacking (B, 3) is correct here; a flat (B*3,) would
                # make tolist() yield ints that cannot unpack.
                out[k] = torch.stack(v)
            else:
                try:
                    out[k] = torch.stack(v)
                except RuntimeError:
                    out[k] = torch.cat(v, dim=0)
        return out

    def _pad_value(self, key: str, tensor) -> int:
        if key == "input_ids":
            return self.pad_token_id
        if key == "labels":
            return -100                              # padding never trained on
        return 0                                     # attention_mask, mm_token_type_ids


def v_ndim(t) -> int:
    return t.dim()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/planner")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", default="out/planner_lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not (data_dir / "manifest.jsonl").exists():
        raise SystemExit(f"No manifest at {data_dir}/manifest.jsonl - run "
                         f"convert_pokedreamer.py first")

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

    dataset = PlannerDataset(data_dir, processor,
                             max_samples=args.max_samples)

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
        remove_unused_columns=False,
        dataloader_num_workers=0,
        warmup_steps=100,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=dataset, processing_class=processor,
                      data_collator=PlannerCollator(
                          processor.tokenizer.pad_token_id))

    print("Starting planner training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.out}/adapter")
    model.save_pretrained(str(Path(args.out) / "adapter"))
    processor.save_pretrained(str(Path(args.out) / "adapter"))
    print("Done.")


if __name__ == "__main__":
    main()