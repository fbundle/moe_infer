# Three-Axis Reasoning Benchmark

Pure-reasoning prompts on three orthogonal axes:

- **Logical** — ZebraLogic (logic-grid CSPs, `grid_mode`)
- **Probabilistic** — CLadder (causal queries, Pearl's hierarchy)
- **Knowledge** — GPQA-Diamond (graduate-level science MC)

Frozen subsets at `data/bench_subsets/{zebralogic,cladder,gpqa_diamond}.jsonl`. Per-example JSONL streams to `data/bench_runs/{model}__{bench}.jsonl` and resumes by ID.

Harnesses (same JSONL schema): `moe_infer/helpers/bench_axes.py` (async-openai + pydantic) and `experiment/bench.go` (sashabaranov + verify-loop with reasoning replay).

## Method

| Axis | Dataset | Subset | Schema | Scoring |
|---|---|---|---|---|
| Logical | `allenai/ZebraLogicBench-private` `grid_mode` | `3*3` + `4*4` (80, easy tier of 1000) | `{header, rows}` | puzzle-wise + cell-wise |
| Probabilistic | `causalNLP/cladder` `full_v1.5_default` | rungs 2 + 3, N=200 | `{answer: "yes"\|"no"}` | exact match |
| Knowledge | `Idavidrein/gpqa` `gpqa_diamond` | full 198 | `{answer: "A"\|"B"\|"C"\|"D"}` | exact letter (shuffled options) |

`no_answer` = wrong (no carve-out).

## Results

| Model | Quant | ZebraLogic | CLadder (R2 / R3) | GPQA-D |
|---|---|---|---|---|
| **deepseek-v4-flash** | API | 100% (64/64) | — | — |
| **VibeThinker-3B** | mixed_4_6 (4.77 bpw) | 70.2% (33/47) | — | — |
| **Qwen3.5-4B** | mlx q4 (~4.5 bpw) | — | — | — |
| **LFM2.5-8B-A1B** (~1B active) | mlx q8 (~8.5 bpw) | — | — | — |
| DeepSeek-R1 | API | 78.7% | 72.3% / 55.1% | 71–73% |
| Claude Sonnet 4 | API | — | 81.2% / 63.4% | — |
| o4-mini-high | API | — | 73.2% / 58.8% | — |
| o1 | API | 81.0% | — | ~78% |
| Claude 3.5 Sonnet | API | 33.4% (all) / 12.4% (hard) | — | 59–65% |
| GPT-4o | API | 31.7% | — | 50–53% |
| GPT-4 (CLadder paper) | API | — | 62% / 70.4% w/ CausalCoT | ~36% |
| Random / human PhD expert | — | ~0% / — | 50% / — | 25% / ~65% |

`max_tokens=40960`, `max_retries=5`, `response_format=json_schema` strict (oMLX accepts but does not constrain). Decode per model card:

| Model | temp | top_p | top_k |
|---|---|---|---|
| VibeThinker-3B | 1.0 | 0.95 | — |
| Qwen3.5-4B | 0.6 | 0.95 | 20 |
| LFM2.5-8B-A1B | 0.2 | 1.0 | — |
| deepseek-v4-flash | 1.0 | 0.95 | — |

## Notes

- **oMLX does not enforce `response_format`.** Direct probe confirms strict `json_schema` is accepted but unconstrained — output discipline is intrinsic to the model.
- **Retry loop is a free win on shape bugs.** Verify-loop replays the prior turn wrapped as `<think>{reasoning_content}</think>{content}` so the model patches malformed JSON instead of re-deriving (re-derivation at temp=1.0 routinely flips correct → wrong). VibeThinker ZL: first-attempt 53% → final 70% (+17pp).
- **No tool calls.** RL-on-text reasoning models (VibeThinker family) read the tool definitions, reason about them in `content`, never emit `<tool_call>`. Chat template still carries the slots but RL trained the emission away.
- **Rung 3 (counterfactual) collapse** is CLadder's diagnostic signal — strong models lose 10–30pp from rung 2.
- **`max_tokens=32k+` is needed for GPQA** — at 16k VibeThinker/DeepSeek truncated ~24%.

## Sources

- GPQA-Diamond: [Artificial Analysis](https://artificialanalysis.ai/evaluations/gpqa-diamond), [llm-stats](https://llm-stats.com/benchmarks/gpqa-diamond)
- ZebraLogic: [arXiv 2502.01100](https://arxiv.org/abs/2502.01100), [AI2 leaderboard](https://huggingface.co/spaces/allenai/ZebraLogic)
- CLadder: [arXiv 2312.04350](https://arxiv.org/abs/2312.04350)
