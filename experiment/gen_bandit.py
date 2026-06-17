"""Generate the bandit subset as JSONL.

Each row: {id, prompt: "", gold: {K, probs}, meta: {K}}

Rewards are sampled at step time (not pre-rolled), seeded by example ID
for reproducibility. T (rounds per game) and other env constants live
in banditBench()'s spec, not in the per-example data.
"""
import json, random
from pathlib import Path

OUT = Path("/Volumes/Hippopotamus/vault/code/moe_infer/data/bench_subsets/bandit.jsonl")
N_EXAMPLES = 200
K_CHOICES = [2, 3, 4, 5]
SEED = 1

rng = random.Random(SEED)
rows = []
for i in range(N_EXAMPLES):
    K = K_CHOICES[i % len(K_CHOICES)]
    probs = [round(rng.uniform(0.1, 0.9), 3) for _ in range(K)]
    rows.append({
        "id": f"bd-{i:03d}",
        "prompt": "",
        "gold": {"K": K, "probs": probs},
        "meta": {"K": K},
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(rows)} examples)")
optimals = [max(r["gold"]["probs"]) for r in rows]
print(f"optimal p — min={min(optimals):.2f} mean={sum(optimals)/len(optimals):.2f} max={max(optimals):.2f}")
