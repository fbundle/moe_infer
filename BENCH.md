# Three-Axis Reasoning Benchmark

Measures model quality on three orthogonal axes:

- **Logical reasoning** — ZebraLogic (logic-grid CSP puzzles, grid mode)
- **Probabilistic reasoning** — CLadder (causal queries, Pearl's hierarchy)
- **Knowledge** — GPQA-Diamond (graduate-level science MC)

No tools — pure-reasoning prompts. JSON-enforced output via `response_format` (strict `json_schema` mode preferred, falls back to `json_object` for endpoints that don't support strict mode, e.g. DeepSeek). Pydantic-validated answers. Per-example dumps in `data/bench_runs/{model}__{bench}.jsonl`.

## Method

| Axis | Dataset | Subset used | Output schema | Scoring |
|---|---|---|---|---|
| Logical | `allenai/ZebraLogicBench-private` `grid_mode` | sizes `3*3` + `4*4` (80 puzzles) | `{header, rows}` | puzzle-wise (all cells match) + cell-wise |
| Probabilistic | `causalNLP/cladder` `full_v1.5_default` | rungs 2 + 3, stratified to N=200 | `{answer: "yes"\|"no"}` | exact match |
| Knowledge | `Idavidrein/gpqa` `gpqa_diamond` | full 198 | `{answer: "A"\|"B"\|"C"\|"D"}` | exact letter match (options shuffled with `--gpqa-seed`) |

If `content` is empty (model truncated mid-thought, or schema-validation fail), we **do not** fall back to scraping `reasoning_content` — that would award random-chance credit on MC. Truncated/invalid → `no_answer`, scored wrong.

Run commands:

```bash
# Cloud (DeepSeek)
python -m moe_infer.helpers.bench_axes \
    --model deepseek-v4-flash \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_TOKEN \
    --benches all --n 200 --concurrency 16 --max-tokens 32768 --request-timeout 600

# Local (Qwen3.5-4B INT4) — use a random subset for tractability
python -m moe_infer.helpers.bench_axes \
    --backend local \
    --model data/Qwen3.5-4B --engine-mode Qwen35DenseFused --quantize-mode int4 \
    --pipeline-cls qwen35_moe --tokenizer-hub data/Qwen3.5-4B/source \
    --benches all --n 30 --sample-seed 1 \
    --concurrency 1 --max-tokens 1024 --request-timeout 300
```

`--sample-seed N` (when `>= 0`) shuffles the dataset before applying `--n`, so a 30-example subset is representative (and reproducible) instead of "first 30."

## Headline results

| Model | Logical (ZebraLogic) | Probabilistic (CLadder) | Knowledge (GPQA-Diamond) |
|---|---|---|---|
| **deepseek-v4-flash** (API, 32k max_tokens) | **98.8%** (79/80) | **83.5%** (167/200) | **83.8%** (166/198) |
| **Qwen3.5-4B INT4** (local, our engine) | **5.0%** (4/80) | **62.5%** (125/200) | _N/A (run stopped early)_ |

## Detailed breakdown

### ZebraLogic — logical reasoning

| Puzzle size | deepseek-v4-flash | Qwen3.5-4B INT4 |
|---|---|---|
| 3×3 puzzle-wise | 100.0% (40/40) | 10.0% (4/40) |
| 4×4 puzzle-wise | 97.5% (39/40) | 0.0% (0/40) |
| **Total puzzle-wise** | **98.8% (79/80)** | **5.0% (4/80)** |
| Cell-wise (all sizes) | **99.5% (1,274/1,280)** | **47.0% (601/1,280)** |

Qwen3.5-4B INT4 emits JSON of the right shape (cell-wise 47% — well above the 25% per-cell random baseline) but only fully solves 5% of puzzles. The 0% on 4×4 vs 10% on 3×3 is the "curse of complexity" already visible at the lower size tier for a 4B-class model.

DeepSeek's single 4×4 miss is essentially saturated at this tier.

### CLadder — probabilistic / causal reasoning

| Rung | Description | deepseek-v4-flash | Qwen3.5-4B INT4 |
|---|---|---|---|
| 2 | Interventional (`do`-operator) | 90.0% (90/100) | 77.0% (77/100) |
| 3 | Counterfactual | 77.0% (77/100) | 48.0% (48/100) |
| **Total** | rungs 2 + 3 | **83.5% (167/200)** | **62.5% (125/200)** |

**Rung-collapse is much steeper on the smaller model** (77 → 48%, a 29 pp drop) than on DeepSeek (90 → 77%, a 13 pp drop). Counterfactual reasoning degrades faster than interventional as model strength drops — exactly the sensitive signal we wanted.

### GPQA-Diamond — knowledge

| Domain | deepseek-v4-flash | Qwen3.5-4B INT4 |
|---|---|---|
| Physics | 95.3% (82/86) | _N/A_ |
| Chemistry | 76.3% (71/93) | _N/A_ |
| Biology | 68.4% (13/19) | _N/A_ |
| **Total** | **83.8% (166/198)** | _N/A — stopped after ~66 of 198_ |

**DeepSeek `max_tokens=32k` recovered the literature-class result** (83.8% vs 88.1% reported on the HF dataset card). The previous 16k run truncated 24% of GPQA questions; at 32k there were **zero truncations** and the score jumped 78.8% → 83.8%. The remaining ~4 pp gap is likely zero-shot prompting (papers use 5-shot CoT) or temperature.

**Qwen3.5-4B GPQA was stopped early** — see the note below on why we'll subset instead of running full sets on local models.

## Runtime summary

| Model | Backend | N | Wall | Effective s/q |
|---|---|---|---|---|
| deepseek-v4-flash (zebralogic) | API, conc=16 | 80 | 151 s | 1.9 s |
| deepseek-v4-flash (cladder) | API, conc=16 | 200 | 140 s | 0.7 s |
| deepseek-v4-flash (gpqa) | API, conc=16 | 198 | 755 s | 3.8 s |
| **deepseek-v4-flash total** | | **478** | **~18 min** | — |
| Qwen3.5-4B INT4 (zebralogic) | Local, conc=1 | 80 | ~28 min | ~21 s |
| Qwen3.5-4B INT4 (cladder) | Local, conc=1 | 200 | ~31 min | ~9 s |
| Qwen3.5-4B INT4 (gpqa) | Local, conc=1 | _stopped at ~66_ | — | ~100 s (cap-bound) |

Cloud is ~25× faster than local in wall time because of API concurrency (16 vs 1); per-query latency is in the same order of magnitude.

## Notes & decisions

- **`max_tokens` asymmetry**: 32k for DeepSeek (reasoning model needs CoT headroom); 1024 for Qwen3.5-4B (non-reasoning, capping the rambling tail). Memory wasn't the constraint — 32k KV on Qwen3.5-4B is ~2 GB, fits easily. Wall time was: GPQA queries at 8.5 tok/s would have taken hours per query at 32k.
- **Local models will use random subsets going forward**. Full-set local runs are expensive: ~3 hr remaining on GPQA when we stopped, all without JSON-enforcement in the engine (the model rambles past `max_tokens` instead of emitting concise JSON). New `--sample-seed N` flag enables reproducible random subsetting across all three benches.
- **Why no reasoning-content fallback**: the previous run scraped letters out of `reasoning_content` when `content` was empty, getting 14/48 truncated GPQA "correct" — that's ~29% which is the random-chance baseline for 4-way MC. Real signal, not luck, requires schema-valid answers in `content`.
- **JSON mode**: DeepSeek's API doesn't support strict `json_schema`, so we use `json_object` mode + client-side pydantic validation. OpenAI proper would use strict mode via `client.chat.completions.parse`.
- **Engine `MAX_SEQ` is now dynamic**: replaced compile-time `MAX_SEQ=4096` with runtime `current_max_seq` on `MetalContext`. Copy-on-grow (doubling) triggers when `pos + n_tokens > current_max_seq`. Observed during Qwen3.5-4B runs: `KV cache grow 4096 → 8192 (16 → 32 MB per layer, 8 full-attn layers)`. Applies to both dense (qwen35_dense) and MoE (qwen35_moe fused_exp{1,2,3,4,5,6}) paths.

## Open work (future runs)

- **Add JSON-grammar constrained decoding** to the local engine. Without it, weaker models (Qwen3.5-4B) don't reliably emit JSON even with a strict system prompt, which inflates wall time and depresses correctness. This is the bottleneck on local-model evaluation, not memory or KV cache.
- **Add few-shot prompts** for GPQA (5-shot CoT is the standard) to close the gap to the 88.1% literature number on DeepSeek.
- **Add ZebraLogic 5×5+ sizes** if/when we want a benchmark that discriminates strong reasoners from each other (the 3×3+4×4 tier is saturating for DeepSeek-class).
