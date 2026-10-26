"""Isolate get_rope_index failure without loading the 8B model.

Stubs self with just config values + the real get_vision_position_ids, then
calls the REAL Qwen2_5_VLModel.get_rope_index with the exact collator batch.
If it passes here, something between Trainer and the model reshapes inputs.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/home/vlad/pokemon-agent")
import torch
from pathlib import Path
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
from train_planner_lora import PlannerDataset, PlannerCollator

proc = AutoProcessor.from_pretrained("/home/vlad/models/Qwen2.5-VL-7B-Instruct")
ds = PlannerDataset(Path("dataset/planner"), proc)
coll = PlannerCollator(proc.tokenizer.pad_token_id)
batch = coll([ds[0], ds[1]])

print("batch shapes:")
for k, v in batch.items():
    print(f"  {k}: {tuple(v.shape)}")

stub = SimpleNamespace(
    config=SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2,
                                      tokens_per_second=25)))
stub.get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids

try:
    pos, deltas = Qwen2_5_VLModel.get_rope_index(
        stub,
        input_ids=batch["input_ids"],
        mm_token_type_ids=batch["mm_token_type_ids"],
        image_grid_thw=batch["image_grid_thw"],
        video_grid_thw=None,
        attention_mask=batch["attention_mask"],
        second_per_grid_ts=None,
    )
    print("get_rope_index OK:", tuple(pos.shape), tuple(deltas.shape))
except Exception as exc:
    import traceback
    traceback.print_exc()
    print("FAILED:", exc)
