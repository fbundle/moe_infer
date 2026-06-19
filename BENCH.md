# Three-Axis Reasoning Benchmark

Three orthogonal axes:

- **Reasoning** — ZebraLogic (logic-grid CSPs, `grid_mode`)
- **Decision under uncertainty** — Synthetic K-armed bandit (forced explore-vs-exploit, no world knowledge)
- **Knowledge** — GPQA-Diamond (graduate-level science MC)

Frozen subsets at `data/bench_subsets/{zebralogic,bandit,gpqa_diamond}.jsonl`. Per-example JSONL streams to `data/bench_runs/{model}__{bench}.jsonl` and resumes by ID.

Harness: `experiment/bench.go` (sashabaranov + verify-loop with reasoning replay).

## Method

| Axis | Dataset | Subset | Schema | Scoring |
|---|---|---|---|---|
| Reasoning | `allenai/ZebraLogicBench-private` `grid_mode` | `3*3` + `4*4` (80, easy tier of 1000) | `{header, rows}` | puzzle-wise + cell-wise |
| Decision | synthetic K-armed Bernoulli bandit | 200 envs, K ∈ {2,3,4,5}, T=30 rounds, rewards sampled per-call from a seeded RNG (no reward table in data) | per turn `{action: int 1..K}` | mean instantaneous regret over T rounds (lower is better); unplayed rounds filled with uniform-random regret |
| Knowledge | `Idavidrein/gpqa` `gpqa_diamond` | full 198 | `{answer: "A"\|"B"\|"C"\|"D"}` | exact letter (shuffled options) |

`no_answer` (model fails to produce valid output after the retry loop's max attempts — `json_parse`, `shape_mismatch`, etc.) = **wrong** on ZL/GPQA; on bandit it's worst-arm regret (so pick the worst possible action).

`error` (distinct from `no_answer`) means a transport/server-side failure (HTTP connection refused, unexpected EOF, status 4xx/5xx from the backend) — the row never reached "the model answered." Error rows are **excluded** from scoring and re-attempted on next queue run (the harness resumes by ID, so deleting error rows from the JSONL queues them up again).

## Results

| Model | Quant | ZebraLogic | Bandit regret ↓ | GPQA-D |
|---|---|---|---|---|
| **deepseek-v4-flash** | API | 100% (80/80) | **0.176 ±0.020** | 83.3% (165/198) |
| **claude-opus-4-7** (claude-cli backend, effort=medium) | API | 100% (80/80) | — | 85.6% (166/194); 4 IDs persistently fail claude-cli (excluded) |
| **VibeThinker-3B** | mixed_4_6 (4.77 bpw) | 66.2% (53/80) (3×3 70%, 4×4 62%) | 0.229 ±0.046 (27/200) — degenerate arm-1 policy | 67% (2/3) partial |
| **VibeThinker-3B** | bf16 (~16 bpw) | 81.25% (65/80) (3×3 82%, 4×4 80%) | skipped (q4 was degenerate; revisit if needed) | 64% (9/14) partial |
| **Qwen3.5-4B (thinking-off)** | mlx q4 (~4.5 bpw) | 3.8% (3/80) (3×3 8%, 4×4 0%) | 0.169 ±0.020 (46/200 valid; 154 oMLX prefill errs pending rerun at conc=1) | 33.3% (66/198) |
| **Qwen3.5-4B (thinking-on)** | mlx q4 (~4.5 bpw) | 86.2% (69/80) (3×3 92%, 4×4 80%) | pending | 55.8% (24/43) partial |
| Claude Opus 4.5 | API | — | — | 87.0% |
| Qwen3-VL-235B-A22B Thinking | open | 97.3% | — | — |
| DeepSeek-V3.2 | open | — | — | ~85% |
| **Gemma-4-E4B-QAT** (4B active / 26B total) | mlx q4 (~4.5 bpw) | on hold | on hold | on hold |
| **LFM2.5-8B-A1B** (~1B active) | mlx q8 (~8.5 bpw) | on hold | on hold | on hold |
| Uniform random | — | ~0% | 0.207 | 25% |
| ε-greedy (ε=0.1) | — | — | 0.120 ±0.004 | — |
| UCB1 | — | — | 0.142 ±0.002 | — |
| Thompson sampling | — | — | 0.107 ±0.002 | — |

## Throughput

Output tokens only. Claude CLI backend reports only final output (no hidden reasoning tokens), so its tok/q is artificially low.

Throughput counts completion_tokens summed across every step (not just the final response), so multi-step axes like bandit show real decode throughput.

### ZebraLogic

| Model | Quant | Tok/q | tok/s |
|---|---|---|---|
| DeepSeek-v4-flash | API | 2253 | 105.1 |
| DeepSeek-v4-pro | API | 2448 | 71.8 |
| Claude Opus 4.7 | API | 780 | 50.3 |
| VibeThinker-3B | mixed_4_6 (4.77 bpw) | 5698 | 32.2 |
| VibeThinker-3B | bf16 (~16 bpw) | 6929 | 9.3 |
| LFM2.5-8B-A1B | mlx q8 (~8.5 bpw) | 6265 | 28.7 (on hold) |
| Gemma-4-E4B-QAT | mlx q4 (~4.5 bpw) | 1390 | 10.8 (on hold, 13/80; too slow) |
| Qwen3.5-4B (thinking-off) | mlx q4 (~4.5 bpw) | 101 | 15.5 |
| Qwen3.5-4B (thinking-on) | mlx q4 (~4.5 bpw) | 7550 | 32.5 |

### Bandit

| Model | Quant | Tok/q | tok/s |
|---|---|---|---|
| DeepSeek-v4-flash | API | 7317 | 79.8 |
| VibeThinker-3B | mixed_4_6 (4.77 bpw) | 4303 | 9.8 (27/200, run terminated) |
| Qwen3.5-4B (thinking-off) | mlx q4 (~4.5 bpw) | 181 | 1.3 (46/200 valid; 154 prefill errs) |

### GPQA-Diamond

| Model | Quant | Tok/q | tok/s |
|---|---|---|---|
| DeepSeek-v4-flash | API | 4333 | 99.4 |
| DeepSeek-v4-pro | API | 6144 | 61.1 |
| Claude Opus 4.7 | API | 1008 | 49.6 (194/194 valid; 4 IDs excluded) |
| Qwen3.5-4B (thinking-off) | mlx q4 (~4.5 bpw) | 10 | 3.3 |
| Qwen3.5-4B (thinking-on) | mlx q4 (~4.5 bpw) | 9960 | 31.5 (partial, 43/198) |
| VibeThinker-3B | bf16 (~16 bpw) | 11162 | 11.6 (partial, 14/198) |

ZL public numbers are the paper's *Overall* (all sizes 2×2 → 6×6). Our 80-row subset is **3×3 + 4×4 only**, which sits in the paper's *Small* and *Medium* tiers — easier than "Overall."

`max_tokens=40960`, `max_retries=5`, `response_format=json_schema` strict (oMLX accepts but does not constrain). Decode per model card:

| Model | temp | top_p | top_k | notes |
|---|---|---|---|---|
| VibeThinker-3B | 1.0 | 0.95 | — | |
| Gemma-4-E4B-QAT | 0.2 | 1.0 | — | |
| Qwen3.5-4B | 0.6 | 0.95 | 20 | thinking mode |
| LFM2.5-8B-A1B | 0.2 | 1.0 | — | |
| deepseek-v4-flash | — | — | — | thinking mode default (silently ignores temp/top_p) |
| claude-opus-4-7 | — | — | — | `claude -p --effort medium --json-schema ...`; uses Claude Code CLI headless, structured_output enforced |

## Notes

- **oMLX does not enforce `response_format`.** Direct probe confirms strict `json_schema` is accepted but unconstrained — output discipline is intrinsic to the model.
- **Retry loop is a free win on shape bugs.** Verify-loop replays the prior turn wrapped as `<think>{reasoning_content}</think>{content}` so the model patches malformed JSON instead of re-deriving (re-derivation at temp=1.0 routinely flips correct → wrong). VibeThinker ZL: first-attempt 51.9% → final 67.5% (+16pp).
- **No tool calls.** RL-on-text reasoning models (VibeThinker family) read the tool definitions, reason about them in `content`, never emit `<tool_call>`. Chat template still carries the slots but RL trained the emission away.
- **Bandit regret = `max(probs) − probs[chosen_arm]`** per round, mean over T=30 rounds. 0 = always picks optimal arm; uniform-random baseline ≈ `max(probs) − mean(probs)`. Agent starts from nothing (no prior, only "each arm has fixed unknown success probability"); each round it picks one arm, observes a 0/1 Bernoulli reward, then decides again. Rounds it never reaches (budget or step cap exhausted) get filled with the uniform-random regret — truncation is penalized, not rewarded. The agent loop is the same `ChatCompletionLoop` used for ZL/GPQA: step either returns intermediate (env observation) or final (rollout complete).
- **`max_tokens=32k+` is needed for GPQA** — at 16k VibeThinker/DeepSeek truncated ~24%.
- **VibeThinker-3B-q4 bandit run terminated at 27/200.** Failure mode is not a prompt issue: the model picked arm 1 for all 30 rounds in 25/27 episodes (93%). Inspecting `reasoning_content` shows the model re-reads each turn's prompt as if it were round 1, ignores the prior `arm X → reward R` history, and burns ~half its CoT on JSON-schema compliance ("must respond with only JSON, ensure no extra text"). Mean regret 0.229 ±0.046 — indistinguishable from uniform-random (0.207). Same axis: DeepSeek-v4-flash explicitly writes per-arm tallies in CoT (`"So far: arm1:1/1, arm2:1/1..."`) and round-robins early — that's why it scores 0.176. Conclusion: the bandit axis isolates sequential-decision-making from chain-of-step reasoning. RL-on-text-reasoning models that weren't trained to update a streaming belief default to a one-shot answer and never explore. Documented; not retrying VibeThinker bandit at higher N.
- **Claude-CLI safety classifier refuses 4 GPQA-Diamond biology questions** (`recypVp2NmPlBKVTp`, `recTs7qzfJs6kfLUK`, `recYt8xx80OTyDsL0`, `rec4L69T0Y1AS4AFS`) — all are pathogen-adjacent virology/immunology MCQs (SARS-CoV-2 molecular biology, transgenic mouse with SARS-CoV-2 receptor, retrovirus diagnostic kit design, rotavirus capsid + B-cell hypermutation). The envelope returns `stop_reason="refusal"` + `is_error=true`. Same prompts pass on the OpenAI-compatible API path (DeepSeek), so this is a Claude Code policy filter, not a question-content artifact. These 4 IDs are excluded from the claude-opus-4-7 denominator (194 valid out of 198).

## Sources

- ZebraLogic: [arXiv 2502.01100](https://arxiv.org/abs/2502.01100), [leaderboard](https://llm-stats.com/benchmarks/zebralogic), [AI2 space](https://huggingface.co/spaces/allenai/ZebraLogic)
- Bandit: synthetic, generated by `experiment/gen_bandit.py` (Lai-Robbins regret baseline; Thompson sampling reference)
- GPQA-Diamond: [Artificial Analysis](https://artificialanalysis.ai/evaluations/gpqa-diamond), [llm-stats](https://llm-stats.com/benchmarks/gpqa-diamond), [vals.ai](https://www.vals.ai/benchmarks/gpqa), [Epoch AI](https://epoch.ai/benchmarks/gpqa-diamond)
