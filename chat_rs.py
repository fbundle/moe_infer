#!/usr/bin/env python3
"""
chat_rs.py — Interactive chat client for Flash-MoE Rust inference server.

Uses the official `openai` Python client to speak the OpenAI-compatible
/v1/chat/completions API. Supports streaming, session persistence, ANSI
markdown rendering, and tool calling.

Usage:
    uv run python chat_rs.py [--port 8000] [--show-think] [--resume <id>]
    uv run python chat_rs.py --sessions
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SESSIONS_DIR = os.path.expanduser("~/.flash-moe/sessions")

# ANSI escape codes for markdown rendering
ANSI_RESET   = "\033[0m"
ANSI_BOLD    = "\033[1m"
ANSI_ITALIC  = "\033[3m"
ANSI_DIM     = "\033[2m"
ANSI_CODE    = "\033[36m"
ANSI_CODEBLK = "\033[48;5;236m\033[38;5;252m"
ANSI_CODEBLK_LINE = "\033[48;5;236m\033[K"
ANSI_HEADER  = "\033[1;34m"
ANSI_YELLOW  = "\033[33m"


def now_ms() -> float:
    return time.time() * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Session persistence
# ─────────────────────────────────────────────────────────────────────────────

def init_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.jsonl")


def session_save_turn(session_id: str, role: str, content: str):
    with open(session_path(session_id), "a") as f:
        entry = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        f.write(entry + "\n")


def session_load(session_id: str) -> int:
    """Replay saved session to screen. Returns number of turns."""
    path = session_path(session_id)
    if not os.path.exists(path):
        return 0

    print(f"[resuming session {session_id}]\n")
    turns = 0
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "user":
                print(f"\033[1m> {content}\033[0m\n")
            elif role == "assistant":
                print(f"{content}\n")
            turns += 1
    if turns > 0:
        print(f"[{turns} turns loaded]\n")
    return turns


def session_list():
    if not os.path.isdir(SESSIONS_DIR):
        print("No sessions found.\n")
        return

    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".jsonl"):
            continue
        sid = fname[:-6]
        path = os.path.join(SESSIONS_DIR, fname)
        lines = 0
        try:
            with open(path) as f:
                for _ in f:
                    lines += 1
        except OSError:
            pass
        mtime = os.path.getmtime(path)
        sessions.append((sid, lines, mtime))

    sessions.sort(key=lambda x: -x[2])
    print("Recent sessions:")
    for sid, lines, _ in sessions:
        print(f"  {sid}  ({lines} turns)")
    if not sessions:
        print("  (none)")
    print()


def generate_session_id() -> str:
    import random
    ts = int(time.time())
    pid = os.getpid()
    rnd = random.randint(1000, 9999)
    return f"chat-{pid}-{ts}{rnd}"


# ─────────────────────────────────────────────────────────────────────────────
# Streaming markdown renderer
# ─────────────────────────────────────────────────────────────────────────────

class MarkdownRenderer:
    """Stateful ANSI markdown renderer for streaming output."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bold = False
        self.italic = False
        self.code_inline = False
        self.code_block = False
        self.skip_lang = False
        self.line_start = True

    def print(self, text: str):
        i = 0
        while i < len(text):
            c = text[i]

            if self.skip_lang:
                if c == '\n':
                    self.skip_lang = False
                    sys.stdout.write(f"{ANSI_CODEBLK}{ANSI_CODEBLK_LINE}\n")
                i += 1
                continue

            if c == '`' and text[i:i+3] == '```':
                if self.code_block:
                    sys.stdout.write(f"{ANSI_RESET}\n")
                    self.code_block = False
                else:
                    self.code_block = True
                    self.skip_lang = True
                i += 3
                continue

            if self.code_block:
                sys.stdout.write(ANSI_CODEBLK)
                if c == '\n':
                    sys.stdout.write(f"{ANSI_CODEBLK_LINE}\n")
                else:
                    sys.stdout.write(c)
                i += 1
                continue

            if c == '`':
                if self.code_inline:
                    sys.stdout.write(ANSI_RESET)
                    self.code_inline = False
                else:
                    sys.stdout.write(ANSI_CODE)
                    self.code_inline = True
                i += 1
                continue

            if self.code_inline:
                sys.stdout.write(c)
                i += 1
                continue

            if self.line_start and c == '#':
                while i < len(text) and text[i] == '#':
                    i += 1
                while i < len(text) and text[i] == ' ':
                    i += 1
                sys.stdout.write(ANSI_HEADER)
                while i < len(text) and text[i] != '\n':
                    sys.stdout.write(text[i])
                    i += 1
                sys.stdout.write(ANSI_RESET)
                if i < len(text) and text[i] == '\n':
                    sys.stdout.write('\n')
                    self.line_start = True
                    i += 1
                continue

            if self.line_start and c in ('-', '*', ' '):
                indent = 0
                peek = i
                while peek < len(text) and text[peek] in (' ', '\t'):
                    indent += 1
                    peek += 1
                if peek < len(text):
                    marker = text[peek]
                    after = text[peek + 1] if peek + 1 < len(text) else '\0'
                    if marker == '-' and after in (' ', '\0', '\t'):
                        depth = indent // 2
                        sys.stdout.write("  " * (depth + 1) + f"{ANSI_YELLOW}•{ANSI_RESET} ")
                        i = peek + 1
                        while i < len(text) and text[i] in (' ', '\t'):
                            i += 1
                        i -= 1
                        self.line_start = False
                        i += 1
                        continue
                    if marker == '*' and after != '*' and after in (' ', '\0', '\t'):
                        depth = indent // 2
                        sys.stdout.write("  " * (depth + 1) + f"{ANSI_YELLOW}•{ANSI_RESET} ")
                        i = peek + 1
                        while i < len(text) and text[i] in (' ', '\t'):
                            i += 1
                        i -= 1
                        self.line_start = False
                        i += 1
                        continue

            if self.line_start and c.isdigit():
                num_start = i
                while i < len(text) and text[i].isdigit():
                    i += 1
                if i < len(text) and text[i] == '.' and (i + 1 >= len(text) or text[i + 1] == ' '):
                    sys.stdout.write(f"  {ANSI_YELLOW}{text[num_start:i]}.{ANSI_RESET} ")
                    if i + 1 < len(text) and text[i + 1] == ' ':
                        i += 2
                    else:
                        i += 1
                    self.line_start = False
                    continue
                i = num_start
                c = text[i]

            if c == '*' and i + 1 < len(text) and text[i + 1] == '*':
                if self.bold:
                    sys.stdout.write(ANSI_RESET)
                    self.bold = False
                else:
                    sys.stdout.write(ANSI_BOLD)
                    self.bold = True
                i += 2
                continue

            if c == '*' and (i + 1 >= len(text) or text[i + 1] != '*'):
                if self.italic:
                    sys.stdout.write(ANSI_RESET)
                    self.italic = False
                else:
                    sys.stdout.write(ANSI_ITALIC)
                    self.italic = True
                i += 1
                continue

            if c == '\n':
                self.line_start = True
            else:
                self.line_start = False

            sys.stdout.write(c)
            i += 1


# ─────────────────────────────────────────────────────────────────────────────
# Chat client (uses openai library)
# ─────────────────────────────────────────────────────────────────────────────

def stream_chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    show_thinking: bool,
) -> tuple[Optional[str], int, float, float]:
    """Stream a chat completion. Returns (full_response, tokens, ttft_ms, gen_time_ms)."""
    md = MarkdownRenderer()
    tokens = 0
    t_start = now_ms()
    t_first = 0.0
    in_think = False
    response_parts: list[str] = []

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue
        content = delta.content
        if not content:
            continue

        tokens += 1
        if not t_first:
            t_first = now_ms()

        if "<think>" in content:
            in_think = True
        if "</think>" in content:
            in_think = False
            continue

        if not in_think:
            response_parts.append(content)

        if in_think and not show_thinking:
            continue
        if in_think:
            sys.stdout.write(f"{ANSI_DIM}{content}{ANSI_RESET}")
        else:
            md.print(content)
        sys.stdout.flush()

    sys.stdout.write(ANSI_RESET)
    t_end = now_ms()
    gen_time = (t_end - t_first) if t_first > 0 else 0

    if tokens > 0 and gen_time > 0:
        tok_s = tokens * 1000.0 / gen_time
        ttft = (t_first - t_start) / 1000.0 if t_first > 0 else 0
        print(f"\n\n[{tokens} tokens, {tok_s:.1f} tok/s, TTFT {ttft:.1f}s]\n")
    elif tokens > 0:
        print("\n")

    full = "".join(response_parts) if response_parts else None
    return full, tokens, t_first - t_start if t_first else 0, gen_time


# ─────────────────────────────────────────────────────────────────────────────
# Tool call handling
# ─────────────────────────────────────────────────────────────────────────────

TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


def extract_tool_command(tc_body: str) -> Optional[str]:
    """Extract bash command from a <tool_call> body."""
    # JSON format: {"name":"bash","arguments":{"command":"..."}}
    cmd_match = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', tc_body)
    if cmd_match:
        return cmd_match.group(1).encode('utf-8').decode('unicode_escape')

    # XML: <arg_value>...</arg_value>
    av_match = re.search(r'<arg_value>(.*?)</arg_value>', tc_body, re.DOTALL)
    if av_match:
        return av_match.group(1).strip()

    # Fallback
    fn_match = re.search(r'bash\s*[>"]?\s*(\S[\s\S]*?)(?:<|\")', tc_body)
    if fn_match:
        return fn_match.group(1).strip()
    return None


def handle_tool_calls(
    client: OpenAI,
    model: str,
    messages: list[dict],
    response: str,
    max_tokens: int,
    session_id: str,
    show_thinking: bool,
    depth: int = 0,
) -> Optional[str]:
    """Handle tool calls in a response. Recurses up to 5 levels deep."""
    if depth >= 5:
        return None

    match = TOOL_CALL_RE.search(response)
    if not match:
        return None

    command = extract_tool_command(match.group(1))
    if not command:
        return None

    print(f"\n{ANSI_YELLOW}$ {command}{ANSI_RESET}")
    sys.stdout.write(f"{ANSI_DIM}[execute? y/n] {ANSI_RESET}")
    sys.stdout.flush()
    ch = sys.stdin.read(1)
    sys.stdin.readline()  # consume rest of line
    if ch.lower() != 'y':
        print(f"{ANSI_DIM}[skipped]{ANSI_RESET}")
        return None

    try:
        result = subprocess.run(command, shell=True, capture_output=True,
                                text=True, timeout=30)
        output = result.stdout
        if result.stderr:
            output += result.stderr
    except subprocess.TimeoutExpired:
        output = "[error: command timed out]"
    except Exception as e:
        output = f"[error: {e}]"

    if output:
        print(f"{ANSI_DIM}{output}{ANSI_RESET}")
        if not output.endswith('\n'):
            print()

    # Send tool response back
    tool_msg = f"<tool_response>\n{output}</tool_response>"
    messages.append({"role": "user", "content": tool_msg})
    session_save_turn(session_id, "user", tool_msg)

    print()
    new_response, _, _, _ = stream_chat(client, model, messages, max_tokens, show_thinking)

    if new_response:
        session_save_turn(session_id, "assistant", new_response)
        if TOOL_CALL_RE.search(new_response):
            return handle_tool_calls(client, model, messages, new_response,
                                     max_tokens, session_id, show_thinking, depth + 1)
    return new_response


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive chat client for Flash-MoE Rust inference server",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--model", type=str, default="flash-moe",
                        help="Model name for API (default: flash-moe)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max response tokens")
    parser.add_argument("--show-think", action="store_true", help="Show <think> blocks (dimmed)")
    parser.add_argument("--resume", type=str, help="Resume a previous session")
    parser.add_argument("--sessions", action="store_true", help="List saved sessions")
    return parser.parse_args()


def health_check(port: int) -> bool:
    """Check whether the server is reachable by listing models."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(f"http://localhost:{port}/v1/models")
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


def main():
    args = parse_args()
    init_sessions_dir()

    if args.sessions:
        session_list()
        return 0

    session_id = args.resume if args.resume else generate_session_id()

    print("=" * 50)
    print("  Flash-MoE Chat (Rust/Metal)")
    print("=" * 50)
    print(f"  Server:  http://localhost:{args.port}/v1")
    print(f"  Model:   {args.model}")
    print(f"  Session: {session_id}{' (resumed)' if args.resume else ''}")
    print()
    print("  Commands: /quit /exit /clear /sessions")
    print("=" * 50)
    print()

    if not health_check(args.port):
        print(f"Server not reachable on port {args.port}.", file=sys.stderr)
        print(f"Start it: cargo run --release -- --serve {args.port}\n", file=sys.stderr)
        return 1

    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="not-needed",  # local server doesn't require auth
    )

    # Resume: load and display history
    if args.resume:
        turns = session_load(session_id)
        if turns == 0:
            print(f"No session found with ID: {session_id}\n")

    # Build initial messages from resumed session
    messages: list[dict] = []
    if args.resume and os.path.exists(session_path(session_id)):
        with open(session_path(session_id)) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    messages.append({"role": entry["role"], "content": entry["content"]})
                except (json.JSONDecodeError, KeyError):
                    pass

    print("Ready to chat.\n")

    # Setup readline
    try:
        import readline
        histfile = os.path.expanduser("~/.flash-moe/history")
        os.makedirs(os.path.dirname(histfile), exist_ok=True)
        try:
            readline.read_history_file(histfile)
        except (FileNotFoundError, PermissionError):
            pass
        readline.set_history_length(500)

        def save_hist():
            try:
                readline.write_history_file(histfile)
            except (OSError, PermissionError):
                pass
    except ImportError:
        readline = None

        def save_hist():
            pass

    # REPL
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue

        if line in ("/quit", "/exit"):
            print("Goodbye.")
            break

        if line == "/clear":
            session_id = generate_session_id()
            messages.clear()
            print(f"[new session: {session_id}]\n")
            continue

        if line == "/sessions":
            session_list()
            continue

        # Save user turn
        session_save_turn(session_id, "user", line)
        messages.append({"role": "user", "content": line})

        print()
        response, tokens, ttft_ms, gen_ms = stream_chat(
            client, args.model, messages, args.max_tokens, args.show_think,
        )

        if response and response.strip():
            session_save_turn(session_id, "assistant", response)
            messages.append({"role": "assistant", "content": response})

        # Handle tool calls
        if response and TOOL_CALL_RE.search(response):
            tc_result = handle_tool_calls(
                client, args.model, messages, response,
                args.max_tokens, session_id, args.show_think,
            )
            if tc_result:
                messages.append({"role": "assistant", "content": tc_result})

    save_hist()
    return 0


if __name__ == "__main__":
    sys.exit(main())
