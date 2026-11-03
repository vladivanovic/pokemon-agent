"""Verify the collator handles MIXED-LENGTH batches (the actual crash case)."""
import sys
sys.path.insert(0, "/home/vlad/pokemon-agent")
from pathlib import Path
from transformers import AutoProcessor
from train_planner_lora import PlannerDataset, PlannerCollator

proc = AutoProcessor.from_pretrained("/home/vlad/models/Qwen2.5-VL-7B-Instruct")
ds = PlannerDataset(Path("dataset/planner"), proc)
coll = PlannerCollator(proc.tokenizer.pad_token_id)

# Find two samples with DIFFERENT input lengths
lens = [ds[i]["input_ids"].shape[0] for i in range(40)]
print("first 40 lengths:", lens)
short, long_ = 0, max(range(40), key=lambda i: lens[i])
if lens[short] == lens[long_]:
    print("WARNING: all same length in first 40")
batch = coll([ds[short], ds[long_]])
print("mixed batch:")
for k, v in batch.items():
    print(f"  {k}: {tuple(v.shape)}")
assert batch["mm_token_type_ids"].dim() == 2, "mm_token_type_ids must be 2-D!"
assert batch["input_ids"].dim() == 2
assert batch["labels"].dim() == 2
print("COLLATOR OK - all sequence tensors 2-D")
