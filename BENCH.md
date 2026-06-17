# Three-Axis Reasoning Benchmark

Three orthogonal axes:

- **Reasoning** — ZebraLogic (logic-grid CSPs, `grid_mode`)
- **Reasoning under uncertainty** — KalshiBench-v2 (resolved prediction-market questions, Brier-scored)
- **Knowledge** — GPQA-Diamond (graduate-level science MC)

Frozen subsets at `data/bench_subsets/{zebralogic,kalshi,gpqa_diamond}.jsonl`. Per-example JSONL streams to `data/bench_runs/{model}__{bench}.jsonl` and resumes by ID.

Harness: `experiment/bench.go` (sashabaranov + verify-loop with reasoning replay).

## Method

| Axis | Dataset | Subset | Schema | Scoring |
|---|---|---|---|---|
| Reasoning | `allenai/ZebraLogicBench-private` `grid_mode` | `3*3` + `4*4` (80, easy tier of 1000) | `{header, rows}` | puzzle-wise + cell-wise |
| Uncertainty | `2084Collective/kalshibench-v2` | 200 sampled | `{probability: float ∈ [0,1]}` | mean Brier (lower is better) + threshold-0.5 accuracy |
| Knowledge | `Idavidrein/gpqa` `gpqa_diamond` | full 198 | `{answer: "A"\|"B"\|"C"\|"D"}` | exact letter (shuffled options) |

`no_answer` = wrong on ZL/GPQA; on Kalshi it's excluded from the mean Brier (so far, n=0 across our runs).

## Results

| Model | Quant | ZebraLogic | Kalshi Brier ↓ | GPQA-D |
|---|---|---|---|---|
| **deepseek-v4-flash** | API | 100% (80/80) | 0.228 | 83.3% (165/198) |
| **deepseek-v4-pro** | API | 100% (80/80) | 0.225 | 82.8% (164/198) |
| **VibeThinker-3B** | mixed_4_6 (4.77 bpw) | 67.5% (52/77) | pending | pending |
| **Qwen3.5-4B** | mlx q4 (~4.5 bpw) | pending | pending | pending |
| **LFM2.5-8B-A1B** (~1B active) | mlx q8 (~8.5 bpw) | pending | pending | pending |
| Claude Opus 4.5 | API | — | 0.227 | 87.0% |
| Qwen3-VL-235B-A22B Thinking | open | 97.3% | — | — |
| DeepSeek-V3.2 | open | — | 0.339 | ~85% |
| Always-0.5 / random baseline | — | ~0% | 0.250 | 25% |

ZL public numbers are the paper's *Overall* (all sizes 2×2 → 6×6). Our 80-row subset is **3×3 + 4×4 only**, which sits in the paper's *Small* and *Medium* tiers — easier than "Overall."

`max_tokens=40960`, `max_retries=5`, `response_format=json_schema` strict (oMLX accepts but does not constrain). Decode per model card:

| Model | temp | top_p | top_k |
|---|---|---|---|
| VibeThinker-3B | 1.0 | 0.95 | — |
| Qwen3.5-4B | 0.6 | 0.95 | 20 |
| LFM2.5-8B-A1B | 0.2 | 1.0 | — |
| deepseek-v4-flash | 1.0 | 0.95 | — |
| deepseek-v4-pro | 1.0 | 0.95 | — |

## Notes

- **oMLX does not enforce `response_format`.** Direct probe confirms strict `json_schema` is accepted but unconstrained — output discipline is intrinsic to the model.
- **Retry loop is a free win on shape bugs.** Verify-loop replays the prior turn wrapped as `<think>{reasoning_content}</think>{content}` so the model patches malformed JSON instead of re-deriving (re-derivation at temp=1.0 routinely flips correct → wrong). VibeThinker ZL: first-attempt 51.9% → final 67.5% (+16pp).
- **No tool calls.** RL-on-text reasoning models (VibeThinker family) read the tool definitions, reason about them in `content`, never emit `<tool_call>`. Chat template still carries the slots but RL trained the emission away.
- **Kalshi Brier = squared error, lower is better** (column header `↓`). 0 = perfect, 0.25 = uninformed (always 0.5), 1.0 = maximally wrong. Always-0.5 baseline = 0.250; superforecasters reach ~0.10; frontier LLMs cluster near 0.22–0.23 — slightly better than uninformed. **Anything > 0.25 is worse than guessing** (e.g. DeepSeek-V3.2 0.339, GPT-5.2-XHigh 0.433).
- **`max_tokens=32k+` is needed for GPQA** — at 16k VibeThinker/DeepSeek truncated ~24%.

## Sources

- ZebraLogic: [arXiv 2502.01100](https://arxiv.org/abs/2502.01100), [leaderboard](https://llm-stats.com/benchmarks/zebralogic), [AI2 space](https://huggingface.co/spaces/allenai/ZebraLogic)
- KalshiBench-v2: [arXiv 2512.16030](https://arxiv.org/abs/2512.16030), [HF dataset](https://huggingface.co/datasets/2084Collective/kalshibench-v2)
- GPQA-Diamond: [Artificial Analysis](https://artificialanalysis.ai/evaluations/gpqa-diamond), [llm-stats](https://llm-stats.com/benchmarks/gpqa-diamond), [vals.ai](https://www.vals.ai/benchmarks/gpqa), [Epoch AI](https://epoch.ai/benchmarks/gpqa-diamond)
