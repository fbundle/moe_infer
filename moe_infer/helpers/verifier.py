"""Sequential ask-and-verify loop with error-feedback retries.

Generic over output format. The verifier OWNS the conversation: callers
provide the starting messages, the verifier appends (assistant, user) pairs
on failure and re-calls. Caller controls the *content* of the user feedback
via `on_failure`; the verifier just chains the calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


class Message(TypedDict):
    """Chat message — minimum subset compatible with OpenAI ChatCompletion API.

    OpenAI's SDK supports richer shapes (multimodal content, tool_call_id,
    tool_calls, name, etc.) but role+content is the universal core. Pass to
    the SDK as-is — TypedDict structurally matches the SDK's accepted dicts.
    """
    role: Role
    content: str


@dataclass
class Completion:
    """Stateless model response. Backends produce these, verifier consumes."""
    content: str
    reasoning_content: str = ""
    finish_reason: FinishReason = "stop"
    parsed: Any = None        # set by strict-mode backend; else None
    tokens_used: int = 0      # completion tokens this call consumed
    extra_mode_info: dict = field(default_factory=dict)


@dataclass
class VerifyLoopOutcome:
    """Number of attempts is just len(history)."""
    comp: Completion | None    # last Completion (None if every call raised)
    parsed: Any                # validated value, or None if loop exhausted
    final_error: str           # last error msg if failed; "" on success
    truncated: bool            # finish_reason == "length" — never retried
    messages: list[Message]    # full final conversation
    history: list[Completion]  # every Completion (one per call) — record-everything


def default_on_failure(comp: Completion | None, error: str) -> str:
    """Default feedback: just the error + a re-emit instruction."""
    return (
        f"That response failed validation: {error}. "
        f"Re-emit ONLY the expected output, no commentary, no markdown fences."
    )


def budget_hint_message(remaining_tokens: int) -> Message:
    """System-role 'sensor' message — fresh clock reading for the model."""
    return Message(
        role="system",
        content=(f"[Budget: {remaining_tokens} tokens remaining for this "
                 f"response. Emit valid output before you run out — "
                 f"truncation = failure.]"),
    )


async def verify_loop(
    *,
    call: Callable[[list[Message]], Awaitable[Completion]],
    validate: Callable[[str], Any],
    initial_messages: list[Message],
    max_retries: int = 0,
    on_failure: Callable[[Completion | None, str], str] = default_on_failure,
    budget_tokens: int | None = None,
) -> VerifyLoopOutcome:
    """Call → validate → (on failure) append (assistant, user) + retry.

    Caller responsibilities:
    - `call(messages)`: send the messages list, return a Completion.
    - `validate(content)`: return parsed value or raise.
    - `initial_messages`: starting conversation (system + user + any context).
    - `on_failure(comp, error_msg)`: build the user-role feedback string.
    - `budget_tokens`: when set, a budget hint is prepended to the
      conversation (visible on EVERY call including the first) AND
      on_failure gets wrapped with `with_budget_hint` so each retry's
      feedback also shows a fresh clock reading.

    Truncation (finish_reason='length') never retries — same prompt would
    truncate identically.
    """
    messages = list(initial_messages)
    remaining = budget_tokens
    out = VerifyLoopOutcome(
        comp=None, parsed=None,
        final_error="", truncated=False, messages=messages, history=[],
    )
    for attempt in range(max_retries + 1):
        # Append a fresh budget reading right before the call. Each call
        # sees the latest at the end of its messages — old budgets
        # naturally fall back into history as the conversation grows.
        if remaining is not None:
            messages.append(budget_hint_message(max(remaining, 0)))

        try:
            out.comp = await call(messages)
            out.parsed = validate(out.comp.content or "")
            out.final_error = ""
        except Exception as e:
            out.parsed = None
            out.final_error = f"{type(e).__name__}: {str(e)[:300]}"

        if out.comp is not None:
            out.history.append(out.comp)
            if remaining is not None:
                remaining -= out.comp.tokens_used

        if out.parsed is not None:
            break  # success
        if out.comp is not None and out.comp.finish_reason == "length":
            out.truncated = True
            out.final_error = out.final_error or "truncated"
            break  # can't fix by retry

        messages.extend([
            Message(role="assistant",
                    content=out.comp.content if out.comp else ""),
            Message(role="user",
                    content=on_failure(out.comp,
                                       out.final_error or "validation failed")),
        ])
    return out
