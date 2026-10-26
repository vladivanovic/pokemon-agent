import sys
sys.path.insert(0, "/home/vlad/pokemon-agent")
import torch
from pathlib import Path
from transformers import AutoProcessor
from train_planner_lora import PlannerDataset, PlannerCollator

proc = AutoProcessor.from_pretrained("/home/vlad/models/Qwen2.5-VL-7B-Instruct")
ds = PlannerDataset(Path("dataset/planner"), proc)
print("dataset size:", len(ds))

# Inspect raw sample shapes
for i in [0, 1, 2]:
    s = ds[i]
    print(f"sample {i}:")
    for k, v in s.items():
        print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")

coll = PlannerCollator(proc.tokenizer.pad_token_id)
batch = coll([ds[0], ds[1]])
print("batch keys:")
for k, v in batch.items():
    print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
