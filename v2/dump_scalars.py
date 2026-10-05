# dump_scalars.py
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv, sys

run = Path(sys.argv[1])          # e.g. runs/poke_ppo_7
ea = EventAccumulator(str(run), size_guidance={"scalars": 100000})
ea.Reload()
tags = ea.Tags()["scalars"]
print(f"{len(tags)} tags")
with open("scalars.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "step", "value"])
    for t in tags:
        for e in ea.Scalars(t):
            w.writerow([t, e.step, e.value])
print("wrote scalars.csv")