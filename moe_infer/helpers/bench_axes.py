"""Three-axis reasoning benchmark via an OpenAI-compatible API.

Axes: ZebraLogic (logical), CLadder (probabilistic), GPQA-Diamond (knowledge).
Pure-reasoning, no tools. JSON output enforced via response_format strict
json_schema (fallback: json_object + client-side pydantic validation).

Per-example streaming dump + resume; sequential verifier-loop retry with
error-feedback turns; optional Pattern D budget hint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from datasets import load_dataset
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ValidationError

from moe_infer.helpers.verifier import Completion, Message, verify_loop


# ── Pydantic output schemas ────────────────────────────────────────────────

class ZebraOutput(BaseModel):
    header: list[str]
    rows: list[list[str]]


class CladderOutput(BaseModel):
    answer: Literal["yes", "no"]


class GpqaOutput(BaseModel):
    answer: Literal["A", "B", "C", "D"]


def _short_display(obj: Any) -> str:
    """Short, human-readable display for per-line logs (gold and parsed)."""
    if obj is None:
        return "None"
    if isinstance(obj, BaseModel):
        if hasattr(obj, "answer"):
            return str(getattr(obj, "answer"))
        if hasattr(obj, "rows"):
            rows = getattr(obj, "rows")
            cols = len(rows[0]) if rows else 0
            return f"<grid {len(rows)}x{cols}>"
        return type(obj).__name__
    if isinstance(obj, dict) and "rows" in obj:
        rows = obj.get("rows", [])
        cols = len(rows[0]) if rows else 0
        return f"<grid {len(rows)}x{cols}>"
    s = str(obj)
    return s[:24] + ("…" if len(s) > 24 else "")


def parse_json_content(content: str, model: type[BaseModel]) -> BaseModel | None:
    """Parse `content` as JSON and validate with the pydantic model.

    The API's `response_format={"type": "json_object"}` mode means `content`
    SHOULD be valid JSON when present. But there are two normal failure modes:
      1. content is empty (reasoning model was truncated mid-thought).
      2. content has a leading ```json … ``` fence (some models still wrap).

    We do NOT fall back to reasoning_content — partial scratchpad text is not
    an answer; we report no-answer instead of awarding random-chance credit.
    """
    if not content:
        return None
    txt = content.strip()
    # Strip optional markdown fence
    if txt.startswith("```"):
        m = re.match(r"```(?:json)?\s*(.*?)\s*```", txt, re.DOTALL)
        if m:
            txt = m.group(1)
    try:
        return model.model_validate_json(txt)
    except ValidationError:
        return None
    except Exception:
        return None


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class Example:
    bench: str
    id: str
    prompt: str
    gold: Any
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    example: Example
    response: str
    parsed: Any
    correct: bool
    elapsed: float
    error: str = ""
    reasoning: str = ""
    extras: dict = field(default_factory=dict)


# ── Benchmark base ─────────────────────────────────────────────────────────

class Bench:
    name: str = ""
    system_prompt: str = ""
    output_model: type[BaseModel] = None  # type: ignore[assignment]

    def load(self, n: int | None) -> list[Example]: raise NotImplementedError
    def parse(self, content: str) -> BaseModel | None:
        return parse_json_content(content, self.output_model)
    def score(self, parsed: BaseModel | None, gold: Any) -> tuple[bool, dict]: raise NotImplementedError
    def report_extras(self, results: list[Result]) -> None: ...

    def load_from_jsonl(self, path: str) -> list[Example]:
        """Load examples from a frozen baseline JSONL (data/bench_subsets/).

        Same shape as a per-axis dump: one JSON record per line with
        `{id, prompt, gold, meta}`. Bypasses dataset+shuffle code paths so
        the example set is fully reproducible across runs even if upstream
        HF dataset rows change.
        """
        out: list[Example] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(Example(
                    bench=self.name, id=d["id"], prompt=d["prompt"],
                    gold=d.get("gold"), meta=d.get("meta", {}),
                ))
        return out


# ── ZebraLogic (logical) ──────────────────────────────────────────────────

class ZebraLogic(Bench):
    name = "zebralogic"
    output_model = ZebraOutput
    system_prompt = (
        "You solve logic-grid puzzles. Respond with a single JSON object "
        "matching this schema (no markdown, no commentary, just JSON):\n"
        '  {"header": ["House", "<attr1>", "<attr2>", ...],\n'
        '   "rows": [["1", "...", ...], ["2", "...", ...], ...]}\n'
        "Use the exact attribute values from the puzzle."
    )

    def __init__(self, sizes: tuple[str, ...] = ("3*3", "4*4"),
                 sample_seed: int | None = None):
        self.sizes = sizes
        self.sample_seed = sample_seed

    def load(self, n: int | None) -> list[Example]:
        ds = load_dataset("allenai/ZebraLogicBench-private", "grid_mode",
                          split="test")
        # Stratify by size so a small subset still has both 3×3 and 4×4
        by_size: dict[str, list[Example]] = {s: [] for s in self.sizes}
        for d in ds:
            if d["size"] not in by_size:
                continue
            by_size[d["size"]].append(Example(
                bench=self.name, id=str(d["id"]),
                prompt=d["puzzle"] + "\n\nSolve and respond as JSON.",
                gold=d["solution"], meta={"size": d["size"]},
            ))
        if self.sample_seed is not None:
            rng = random.Random(self.sample_seed)
            for items in by_size.values():
                rng.shuffle(items)
        if n is None:
            return [x for items in by_size.values() for x in items]
        # Distribute n across sizes
        base, rem = divmod(n, len(by_size))
        out: list[Example] = []
        for i, s in enumerate(by_size):
            cap = base + (1 if i < rem else 0)
            out.extend(by_size[s][:cap])
        return out

    def score(self, parsed: BaseModel | None, gold: Any) -> tuple[bool, dict]:
        gold_rows = gold["rows"]
        n_total = sum(len(r) for r in gold_rows)
        if not isinstance(parsed, ZebraOutput):
            return False, {"cell_correct": 0, "cell_total": n_total}
        correct = 0
        for gr, pr in zip(gold_rows, parsed.rows):
            for gc, pc in zip(gr, pr):
                if str(gc).strip().lower() == str(pc).strip().lower():
                    correct += 1
        all_match = (correct == n_total and len(gold_rows) == len(parsed.rows))
        return all_match, {"cell_correct": correct, "cell_total": n_total}

    def report_extras(self, results: list[Result]) -> None:
        tot = sum(r.extras.get("cell_total", 0) for r in results)
        ok = sum(r.extras.get("cell_correct", 0) for r in results)
        if tot:
            print(f"  cell-wise = {ok/tot:.1%}  ({ok}/{tot})")
        by_size: dict[str, list[Result]] = defaultdict(list)
        for r in results:
            by_size[r.example.meta.get("size", "?")].append(r)
        for sz, rs in sorted(by_size.items()):
            ok = sum(1 for r in rs if r.correct)
            print(f"  {sz:>5}: puzzle-wise {ok}/{len(rs)} = {ok/len(rs):.1%}")


# ── CLadder (probabilistic / causal) ──────────────────────────────────────

class CLadder(Bench):
    name = "cladder"
    output_model = CladderOutput
    system_prompt = (
        "Answer the causal-reasoning question. Respond with a single JSON "
        'object: {"answer": "yes"} or {"answer": "no"}.'
    )

    RUNG_NAMES = {1: "associational", 2: "interventional", 3: "counterfactual"}

    def __init__(self, rungs: tuple[int, ...] = (2, 3),
                 per_rung: int | None = None,
                 sample_seed: int | None = None):
        self.rungs = rungs
        self.per_rung = per_rung
        self.sample_seed = sample_seed

    def load(self, n: int | None) -> list[Example]:
        ds = load_dataset("causalNLP/cladder", split="full_v1.5_default")
        by_rung: dict[int, list] = {r: [] for r in self.rungs}
        for d in ds:
            if d["rung"] in by_rung:
                by_rung[d["rung"]].append(d)

        if self.sample_seed is not None:
            rng = random.Random(self.sample_seed)
            for items in by_rung.values():
                rng.shuffle(items)

        # Stratified sample: aim for ~equal counts per rung that sum to n.
        rung_caps: dict[int, int | None] = {r: self.per_rung for r in self.rungs}
        if self.per_rung is None and n is not None:
            base, rem = divmod(n, len(self.rungs))
            for i, r in enumerate(self.rungs):
                rung_caps[r] = base + (1 if i < rem else 0)

        ex_list: list[Example] = []
        for r in self.rungs:
            items = by_rung[r]
            cap = rung_caps[r]
            if cap is not None:
                items = items[:cap]
            for d in items:
                ex_list.append(Example(
                    bench=self.name,
                    id=str(d["id"]),
                    prompt=d["prompt"],
                    gold=d["label"].strip().lower(),
                    meta={
                        "rung": d["rung"],
                        "query_type": d["query_type"],
                        "difficulty": d.get("question_property", ""),
                    },
                ))
        return ex_list

    def score(self, parsed: BaseModel | None, gold: Any) -> tuple[bool, dict]:
        if not isinstance(parsed, CladderOutput):
            return False, {}
        return (parsed.answer == gold), {}

    def report_extras(self, results: list[Result]) -> None:
        by_rung: dict[int, list[Result]] = defaultdict(list)
        for r in results:
            by_rung[r.example.meta.get("rung", 0)].append(r)
        for rung in sorted(by_rung):
            rs = by_rung[rung]
            ok = sum(1 for r in rs if r.correct)
            name = self.RUNG_NAMES.get(rung, str(rung))
            print(f"  rung-{rung} ({name}): {ok}/{len(rs)} = {ok/len(rs):.1%}")


# ── GPQA-Diamond (knowledge) ──────────────────────────────────────────────

class GPQADiamond(Bench):
    name = "gpqa_diamond"
    output_model = GpqaOutput
    system_prompt = (
        "Answer the multiple-choice question. Respond with a single JSON "
        'object: {"answer": "A"} or "B", "C", "D".'
    )

    def __init__(self, shuffle_seed: int = 0,
                 sample_seed: int | None = None):
        self.shuffle_seed = shuffle_seed
        self.sample_seed = sample_seed

    def load(self, n: int | None) -> list[Example]:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rng = random.Random(self.shuffle_seed)
        letters = ["A", "B", "C", "D"]
        ex_list: list[Example] = []
        for d in ds:
            opts = [
                ("correct", str(d["Correct Answer"]).strip()),
                ("inc_1", str(d["Incorrect Answer 1"]).strip()),
                ("inc_2", str(d["Incorrect Answer 2"]).strip()),
                ("inc_3", str(d["Incorrect Answer 3"]).strip()),
            ]
            rng.shuffle(opts)
            correct_letter = "?"
            block = ""
            for i, (kind, text) in enumerate(opts):
                block += f"{letters[i]}) {text}\n"
                if kind == "correct":
                    correct_letter = letters[i]
            prompt = (
                f"{d['Question'].strip()}\n\n"
                f"Options:\n{block}\n"
                f"Choose the single best letter."
            )
            ex_list.append(Example(
                bench=self.name,
                id=str(d.get("Record ID", "")),
                prompt=prompt,
                gold=correct_letter,
                meta={
                    "domain": d.get("High-level domain", ""),
                    "subdomain": d.get("Subdomain", ""),
                },
            ))
        if self.sample_seed is not None:
            random.Random(self.sample_seed).shuffle(ex_list)
        if n is not None:
            ex_list = ex_list[:n]
        return ex_list

    def score(self, parsed: BaseModel | None, gold: Any) -> tuple[bool, dict]:
        if not isinstance(parsed, GpqaOutput):
            return False, {}
        return (parsed.answer == gold), {}

    def report_extras(self, results: list[Result]) -> None:
        by_dom: dict[str, list[Result]] = defaultdict(list)
        for r in results:
            by_dom[r.example.meta.get("domain", "?")].append(r)
        for dom, rs in sorted(by_dom.items()):
            ok = sum(1 for r in rs if r.correct)
            print(f"  {dom:>20}: {ok}/{len(rs)} = {ok/len(rs):.1%}")


# ── Async runner ──────────────────────────────────────────────────────────

# ── Backend: OpenAI-compatible API ────────────────────────────────────────

class OpenAIBackend:
    """Stateless wrapper over the OpenAI-compatible chat completion API.

    Given `messages` and a response schema, call the SDK and return a
    `Completion`. Tries strict json_schema first; falls back permanently to
    json_object when the endpoint refuses (e.g. DeepSeek).
    """

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model
        # Cache: True = strict OK, False = use json_object, None = unknown.
        self.strict_ok: bool | None = None

    async def complete(
        self, *, messages: list[Message],
        response_format: type[BaseModel],
        client_parse,             # fn(content: str) -> parsed value or raises
        max_tokens: int, temperature: float, top_p: float = 1.0,
    ) -> Completion:
        if self.strict_ok is not False:
            try:
                resp = await self.client.chat.completions.parse(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                    response_format=response_format,
                )
                self.strict_ok = True
                msg = resp.choices[0].message
                return Completion(
                    content=msg.content or "",
                    reasoning_content=getattr(msg, "reasoning_content", "") or "",
                    finish_reason=resp.choices[0].finish_reason or "stop",
                    parsed=msg.parsed,
                    tokens_used=(resp.usage.completion_tokens
                                 if resp.usage else 0),
                    extra_mode_info={"mode": "strict"},
                )
            except BadRequestError as e:
                if "response_format" in str(e).lower() or "unavailable" in str(e).lower():
                    self.strict_ok = False
                else:
                    raise
        # json_object fallback
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, top_p=top_p,
            response_format={"type": "json_object"},
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        try:
            parsed = client_parse(content)
        except Exception:
            parsed = None
        return Completion(
            content=content,
            reasoning_content=getattr(msg, "reasoning_content", "") or "",
            finish_reason=resp.choices[0].finish_reason or "stop",
            parsed=parsed,
            tokens_used=(resp.usage.completion_tokens
                         if resp.usage else 0),
            extra_mode_info={"mode": "json_object"},
        )


# ── Backend: local Pipeline (single-GPU, serialized) ──────────────────────

class LocalPipelineBackend:
    """Runs against our in-process Pipeline (single GPU, no batched inference).

    Concurrency must be 1 (or wrapped in the lock below) because the Pipeline
    holds a single KV cache + Metal context. The harness's outer semaphore
    enforces this when --concurrency 1, but we also hold an internal lock
    in case the user forgets.
    """

    def __init__(self, pipe, sampler_temperature: float = 0.0):
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.default_temperature = sampler_temperature

    async def complete(self, bench: Bench, ex: Example,
                       max_tokens: int, temperature: float) -> Completion:
        async with self.lock:
            # Run synchronously on the asyncio thread — the Rust Cache is
            # `unsendable` (pyo3) and would panic if dispatched to a worker
            # thread. Concurrency for the local backend is forced to 1, so
            # blocking the loop briefly is fine.
            text = self._chat_sync(
                bench.system_prompt, ex.prompt, max_tokens, temperature,
            )
        return Completion(
            content=text,
            parsed=bench.parse(text),
            extra_mode_info={"mode": "local_pipeline"},
        )

    def _chat_sync(self, system: str, user: str,
                   max_tokens: int, temperature: float) -> str:
        self.pipe.reset()
        # Inject a system message before the user's. Pipeline's chat()
        # apply_chat_template the full _messages list on the first turn,
        # so this puts the system prompt into the prompt cleanly.
        self.pipe._messages = [{"role": "system", "content": system}]
        return self.pipe.chat(
            user, max_tokens=max_tokens, temperature=temperature,
        )


# ── Per-example runner ────────────────────────────────────────────────────

def _budget_skipped_result(ex: Example) -> Result:
    return Result(
        example=ex, response="", parsed=None,
        correct=False, elapsed=0.0, error="",
        extras={
            "no_answer": True,
            "budget_skipped": True,
            "finish": "skip",
            "len_content": 0,
            "len_reasoning": 0,
        },
    )


async def _run_one(
    backend, ex: Example, bench: Bench,
    semaphore: asyncio.Semaphore, max_tokens: int, temperature: float,
    request_timeout: float,
    deadline: float | None,
    top_p: float = 1.0,
    max_retries: int = 0,
    budget_hint: bool = False,
) -> Result:
    async with semaphore:
        # Deadline check INSIDE critical section — checking from the outer
        # loop has a race: asyncio releases the semaphore before the flag
        # can flip. Bounded overshoot: only the task already past this
        # check at deadline time runs; later tasks skip cheaply.
        if deadline is not None and time.perf_counter() >= deadline:
            return _budget_skipped_result(ex)
        t0 = time.perf_counter()

        # ── Build the conversation in one place ──
        initial_messages: list[Message] = [
            Message(role="system", content=bench.system_prompt),
            Message(role="user", content=ex.prompt),
        ]

        async def call(messages):
            return await asyncio.wait_for(
                backend.complete(
                    messages=messages,
                    response_format=bench.output_model,
                    client_parse=bench.parse,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                ),
                timeout=request_timeout,
            )

        outcome = await verify_loop(
            call=call,
            validate=lambda content: bench.output_model.model_validate_json(content),
            initial_messages=initial_messages,
            max_retries=max_retries,
            budget_tokens=max_tokens if budget_hint else None,
        )
        elapsed = time.perf_counter() - t0

        if outcome.comp is None:
            return Result(
                example=ex, response="", parsed=None,
                correct=False, elapsed=elapsed, error=outcome.final_error,
                extras={"attempts": len(outcome.history)},
            )

        # Prefer verify_loop's parsed (last successful validate); else fall
        # back to the strict-mode SDK's parsed value if it populated one.
        final_parsed = (outcome.parsed if outcome.parsed is not None
                        else outcome.comp.parsed)
        correct, extras = bench.score(final_parsed, ex.gold)
        extras = {
            **extras,
            "finish": outcome.comp.finish_reason,
            "len_content": len(outcome.comp.content),
            "len_reasoning": len(outcome.comp.reasoning_content),
            "no_answer": final_parsed is None,
            "attempts": len(outcome.history),
            "tokens_used": outcome.comp.tokens_used,
            "truncated": outcome.truncated,
            # Per-attempt full record so failures can be replayed/inspected.
            "history": [
                {
                    "content": c.content,
                    "reasoning_content": c.reasoning_content,
                    "finish_reason": c.finish_reason,
                    "tokens_used": c.tokens_used,
                    **c.extra_mode_info,
                }
                for c in outcome.history
            ],
            # Final conversation that was sent (with budget hints + retry turns).
            "messages": outcome.messages,
            **outcome.comp.extra_mode_info,
        }
        return Result(
            example=ex, response=outcome.comp.content,
            reasoning=outcome.comp.reasoning_content,
            parsed=final_parsed,
            correct=correct, elapsed=elapsed, extras=extras,
        )


def _serialize_result(r: Result) -> dict:
    parsed_dump = (r.parsed.model_dump()
                   if isinstance(r.parsed, BaseModel) else r.parsed)
    return {
        "id": r.example.id,
        "prompt": r.example.prompt,  # full prompt — record everything
        "gold": r.example.gold,
        "response": r.response,
        "reasoning": r.reasoning,
        "parsed": parsed_dump,
        "correct": r.correct,
        "elapsed": r.elapsed,
        "error": r.error,
        "meta": r.example.meta,
        "extras": r.extras,
    }


def _result_from_record(d: dict, bench_name: str) -> Result:
    """Reconstruct a Result from a streaming-dump JSONL record.

    `parsed` round-trips as a dict (or None); that's fine for stats —
    we already stored `correct`. Score/parse aren't re-run on resume.
    """
    ex = Example(
        bench=bench_name, id=d["id"],
        prompt=d.get("prompt", ""), gold=d.get("gold"),
        meta=d.get("meta", {}),
    )
    return Result(
        example=ex,
        response=d.get("response", ""),
        reasoning=d.get("reasoning", ""),
        parsed=d.get("parsed"),
        correct=bool(d.get("correct", False)),
        elapsed=float(d.get("elapsed", 0.0)),
        error=d.get("error", ""),
        extras=d.get("extras", {}),
    )


def _mark_for(r: Result) -> str:
    if r.extras.get("budget_skipped"):
        return "."
    if r.error:
        return "E"
    if r.extras.get("no_answer"):
        return "?"
    return "+" if r.correct else "-"


def _print_example_line(r: Result, done: int, total: int) -> None:
    if r.extras.get("budget_skipped"):
        return
    mark = _mark_for(r)
    gold_s = _short_display(r.example.gold)
    parsed_s = _short_display(r.parsed)
    finish = (r.extras.get("finish") or "?")[:6]
    lc = r.extras.get("len_content", 0)
    lr = r.extras.get("len_reasoning", 0)
    suffix = (f"  err={r.error[:80]}" if r.error else "")
    print(f"  {mark} {done:>3}/{total}  id={r.example.id:>12}  "
          f"{r.elapsed:>5.1f}s  fin={finish:<6} c={lc:<4} r={lr:<5}  "
          f"gold={gold_s:<22}  parsed={parsed_s:<22}{suffix}",
          flush=True)


async def run_bench(
    backend, model_label: str, bench: Bench, n: int | None,
    concurrency: int, max_tokens: int, temperature: float,
    request_timeout: float, dump_dir: str | None,
    time_budget_sec: float = 0.0,
    subsets_dir: str | None = None,
    top_p: float = 1.0,
    max_retries: int = 0,
    budget_hint: bool = False,
) -> tuple[float, list[Result]]:
    # Prefer the frozen baseline subset when present — keeps example IDs
    # stable across runs (and across upstream HF dataset edits).
    subset_path = None
    if subsets_dir:
        candidate = os.path.join(subsets_dir, f"{bench.name}.jsonl")
        if os.path.exists(candidate):
            subset_path = candidate
    if subset_path:
        examples = bench.load_from_jsonl(subset_path)
        msg = (f"[{bench.name}] using frozen subset {subset_path} "
               f"({len(examples)} examples). --zebra-sizes, --cladder-rungs, "
               f"--gpqa-seed, --sample-seed are ignored.")
        if n is not None and n < len(examples):
            examples = examples[:n]
            msg += f" Capped to first {n}."
        print(msg, flush=True)
    else:
        examples = bench.load(n)

    # ── Resume: load any prior streaming dump for this (model, bench) ──
    dump_path: str | None = None
    completed: dict[str, Result] = {}
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(
            dump_dir, f"{model_label.replace('/', '_')}__{bench.name}.jsonl",
        )
        if os.path.exists(dump_path):
            with open(dump_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    completed[rec["id"]] = _result_from_record(rec, bench.name)

    resumed = [completed[ex.id] for ex in examples if ex.id in completed]
    todo = [ex for ex in examples if ex.id not in completed]

    print(f"\n{'='*64}\n[{bench.name}] model={model_label}  n={len(examples)}  "
          f"resumed={len(resumed)}  todo={len(todo)}  "
          f"concurrency={concurrency}  budget={time_budget_sec:.0f}s\n{'='*64}",
          flush=True)
    # Replay resumed examples so log + progress watcher see them.
    for i, r in enumerate(resumed, start=1):
        _print_example_line(r, i, len(examples))

    if not examples:
        print(f"[{bench.name}] no examples loaded", flush=True)
        return 0.0, []

    # ── Open dump in append mode for streaming writes ──
    dump_f = open(dump_path, "a") if dump_path else None

    sem = asyncio.Semaphore(concurrency)
    t_start = time.perf_counter()
    deadline = (t_start + time_budget_sec) if time_budget_sec > 0 else None
    running_tasks = [
        asyncio.create_task(_run_one(backend, ex, bench, sem,
                                     max_tokens, temperature, request_timeout,
                                     deadline, top_p=top_p,
                                     max_retries=max_retries,
                                     budget_hint=budget_hint))
        for ex in todo
    ]

    fresh: list[Result] = []
    done = len(resumed)
    budget_announced = False
    try:
        for fut in asyncio.as_completed(running_tasks):
            try:
                r = await fut
            except asyncio.CancelledError:
                continue
            done += 1
            fresh.append(r)
            # Stream-dump immediately (before printing or computing stats)
            if dump_f is not None and not r.extras.get("budget_skipped"):
                dump_f.write(json.dumps(_serialize_result(r), default=str) + "\n")
                dump_f.flush()
            _print_example_line(r, done, len(examples))
            if (not budget_announced and time_budget_sec > 0
                    and r.extras.get("budget_skipped")):
                budget_announced = True
                elapsed = time.perf_counter() - t_start
                n_pending = sum(1 for t in running_tasks if not t.done())
                print(f"  [{bench.name}] budget {time_budget_sec:.0f}s reached "
                      f"after {done - 1} completed; {n_pending} pending "
                      f"will short-circuit (elapsed={elapsed:.0f}s)",
                      flush=True)
    finally:
        if dump_f is not None:
            dump_f.close()

    wall = time.perf_counter() - t_start
    results = resumed + fresh

    attempted = [r for r in results if not r.extras.get("budget_skipped")]
    n_skipped = len(results) - len(attempted)
    n_correct = sum(1 for r in attempted if r.correct)
    n_error = sum(1 for r in attempted if r.error)
    n_no_ans = sum(1 for r in attempted if r.extras.get("no_answer"))
    n_trunc = sum(1 for r in attempted if r.extras.get("finish") == "length")
    acc = n_correct / max(1, len(attempted))
    s_per_q = wall / max(1, len(attempted))
    print(f"\n[{bench.name}] accuracy = {acc:.1%}  ({n_correct}/{len(attempted)})  "
          f"no_answer={n_no_ans}  truncated={n_trunc}  errors={n_error}  "
          f"budget_skipped={n_skipped}  wall={wall:.0f}s  ({s_per_q:.1f} s/q)",
          flush=True)
    answered = len(attempted) - n_no_ans
    if answered > 0:
        clean_acc = n_correct / answered
        print(f"  clean accuracy (answered only) = {clean_acc:.1%}  "
              f"({n_correct}/{answered})", flush=True)
    bench.report_extras(attempted)

    if dump_path:
        print(f"  dumped → {dump_path}  (streamed; {len(resumed)} resumed + "
              f"{len(fresh)} fresh)")

    return acc, results


# ── CLI ────────────────────────────────────────────────────────────────────

def _normalize_base_url(url: str) -> str:
    """Some users paste the full endpoint URL (incl. /chat/completions).
    The OpenAI SDK appends paths itself; strip the suffix if present."""
    url = url.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


def _build_backend(args: argparse.Namespace) -> tuple[Any, str]:
    """Return (backend, model_label).

    `model_label` is used for dump filenames and the summary header.
    """
    if args.backend == "openai":
        api_key = args.api_key or os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"no API key — set env var {args.api_key_env} or pass --api-key"
            )
        base_url = _normalize_base_url(args.base_url)
        print(f"[backend] openai  base_url={base_url}  model={args.model}")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return OpenAIBackend(client, args.model), args.model
    elif args.backend == "local":
        # Pick the matching Pipeline subclass. For Qwen3.5-4B-dense the base
        # Pipeline works; for MoE models we need the qwen35_moe pipeline so
        # its eos_ids / chat template are right.
        if args.pipeline_cls == "base":
            from moe_infer.pipeline import Pipeline as PipeCls
        elif args.pipeline_cls == "qwen35_moe":
            from moe_infer.qwen35_moe.pipeline import Qwen35MoEPipeline as PipeCls
        else:
            raise SystemExit(f"unknown --pipeline-cls {args.pipeline_cls}")

        kwargs: dict[str, Any] = {
            "mode": args.engine_mode,
            "quantize_mode": args.quantize_mode,
            "expert_cache_count": args.expert_cache_count,
        }
        if args.tokenizer_hub:
            kwargs["hub"] = args.tokenizer_hub
        print(f"[backend] local  model_path={args.model}  engine_mode={args.engine_mode}  "
              f"quantize_mode={args.quantize_mode}  pipeline_cls={args.pipeline_cls}")
        pipe = PipeCls(args.model, **kwargs)
        label = os.path.basename(args.model.rstrip("/"))
        return LocalPipelineBackend(pipe), label
    else:
        raise SystemExit(f"unknown --backend {args.backend}")


async def main_async(args: argparse.Namespace) -> None:
    backend, model_label = _build_backend(args)

    seed = args.sample_seed if args.sample_seed >= 0 else None
    bench_map: dict[str, Bench] = {
        "zebralogic": ZebraLogic(
            sizes=tuple(args.zebra_sizes.split(",")),
            sample_seed=seed,
        ),
        "cladder": CLadder(
            rungs=tuple(int(r) for r in args.cladder_rungs.split(",")),
            sample_seed=seed,
        ),
        "gpqa": GPQADiamond(
            shuffle_seed=args.gpqa_seed,
            sample_seed=seed,
        ),
    }

    benches = list(bench_map) if args.benches == "all" else args.benches.split(",")
    unknown = [b for b in benches if b not in bench_map]
    if unknown:
        raise SystemExit(f"unknown benchmarks: {unknown}. valid={list(bench_map)}")

    summary: dict[str, float] = {}
    for b in benches:
        acc, _ = await run_bench(
            backend, model_label, bench_map[b], args.n,
            args.concurrency, args.max_tokens, args.temperature,
            args.request_timeout, args.dump_dir,
            time_budget_sec=args.time_budget_sec,
            subsets_dir=args.subsets_dir,
            top_p=args.top_p,
            max_retries=args.max_retries,
            budget_hint=args.budget_hint,
        )
        summary[b] = acc

    print("\n" + "=" * 64)
    print(f"SUMMARY  model={model_label}")
    print("=" * 64)
    for b, acc in summary.items():
        print(f"  {b:<14} {acc:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()

    # ── Backend selection ──
    ap.add_argument("--backend", default="openai", choices=["openai", "local"],
                    help="'openai' = any OpenAI-compatible API. "
                    "'local' = our in-process Pipeline.")

    # ── OpenAI backend args ──
    ap.add_argument("--model", default="deepseek-v4-flash",
                    help="OpenAI model name OR local model directory path")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_TOKEN",
                    help="env var to read the API key from")
    ap.add_argument("--api-key", default=None,
                    help="literal API key (overrides --api-key-env). "
                    "Avoid in shell history — prefer the env var.")

    # ── Local backend args ──
    ap.add_argument("--engine-mode", default="Qwen35DenseFused",
                    help="local: Rust engine kind (e.g. Qwen35DenseFused, "
                    "Qwen35MoEFusedExp5)")
    ap.add_argument("--quantize-mode", default="int4",
                    help="local: 'int4' or 'bq4' — chooses model_{q}/ subdir")
    ap.add_argument("--pipeline-cls", default="base",
                    choices=["base", "qwen35_moe"],
                    help="local: pick Pipeline class (chat template / eos)")
    ap.add_argument("--tokenizer-hub", default=None,
                    help="local: path to a HF directory holding tokenizer.json "
                    "(e.g. data/Qwen3.5-4B/source). Otherwise auto-discovered "
                    "from <model>/tokenizer/")
    ap.add_argument("--expert-cache-count", type=int, default=0)

    # ── Benchmark / run args ──
    ap.add_argument("--benches", default="all",
                    help="comma list: zebralogic,cladder,gpqa | 'all'")
    ap.add_argument("--n", type=int, default=None,
                    help="cap on examples per benchmark (None = full / default)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="for --backend local, set to 1 (the Pipeline is "
                    "single-GPU and we serialize internally anyway)")
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="reasoning models need a lot of headroom (32k+)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="nucleus sampling cutoff. Default 1.0 (no nucleus). "
                    "VibeThinker-3B card recommends 0.95.")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="resample on parse-fail (when finish=stop, NOT length). "
                    "Default 5 — recovers most stochastic parse-fails at temp>0. "
                    "Truncated (finish=length) samples are never retried because "
                    "the same prompt would truncate identically.")
    ap.add_argument("--budget-hint", action="store_true",
                    help="Append a token-budget hint to the system prompt "
                    "(Pattern A): 'you have at most N tokens, emit JSON before "
                    "you run out.' Helps reasoning models avoid finish=length "
                    "truncation. Off by default.")
    ap.add_argument("--request-timeout", type=float, default=600.0,
                    help="reasoning + local can be slow; set 600s")
    ap.add_argument("--zebra-sizes", default="3*3,4*4")
    ap.add_argument("--cladder-rungs", default="2,3")
    ap.add_argument("--gpqa-seed", type=int, default=0,
                    help="seed for GPQA MC-option shuffling (fixed regardless "
                    "of --sample-seed, since the task itself randomizes options)")
    ap.add_argument("--sample-seed", type=int, default=-1,
                    help="if >=0, shuffle examples before applying --n. Use for "
                    "representative subsets on local-model runs (e.g. --n 30 "
                    "--sample-seed 1 picks 30 random examples reproducibly). "
                    "Default -1 = no shuffle (deterministic 'first N').")
    ap.add_argument("--time-budget-sec", type=float, default=0.0,
                    help="per-benchmark wall-time cap (s). After this, in-flight "
                    "results are kept and pending tasks are cancelled. 0 = no cap.")
    ap.add_argument("--dump-dir", default="data/bench_runs")
    ap.add_argument("--subsets-dir", default="data/bench_subsets",
                    help="frozen baseline subset dir. If {axis}.jsonl exists "
                    "there, it overrides dataset loading; --n, --zebra-sizes, "
                    "--cladder-rungs, --gpqa-seed, --sample-seed are ignored. "
                    "Pass empty string to disable.")

    args = ap.parse_args()
    if args.subsets_dir == "":
        args.subsets_dir = None
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
