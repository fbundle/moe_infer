#!/usr/bin/env python3
"""Interactive chat client for Flash-MoE inference server.

Usage:
  python helpers/chat.py                     # connect to localhost:8000
  python helpers/chat.py --port 8080          # custom port
  python helpers/chat.py --resume <id>        # resume a session
"""
import argparse
import json
import os
import re
import readline
import sys
import time
import urllib.request
import urllib.error

SESSIONS_DIR = os.path.expanduser("~/.flash-moe/sessions")

# BPE artifacts — the tokenizer leaves these in decoded output
BPE_CLEANUP = [
    ("Ġ", " "),    # GPT-2 / Qwen space token
    ("Ċ", "\n"),   # newline artifact
    ("âĢĶ", "—"),  # emdash
    ("âĢĵ", "–"),  # endash
    ("âĢľ", "''"),
    ("âĢĻ", "\""),
    ("ĉ", ""),     # stray continuation byte
]


def clean_text(text: str) -> str:
    """Clean BPE artifacts from decoded text."""
    for a, b in BPE_CLEANUP:
        text = text.replace(a, b)
    return text


def load_sessions():
    """List saved sessions."""
    if not os.path.isdir(SESSIONS_DIR):
        return {}
    sessions = {}
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".jsonl"):
            sid = fname[:-6]
            path = os.path.join(SESSIONS_DIR, fname)
            sessions[sid] = path
    return sessions


def stream_chat(host: str, port: int, prompt: str, max_tokens: int,
                session_id: str, show_think: bool = False) -> str:
    """Send chat request to infer.m server and stream SSE response.
    Returns the full response text.
    """
    body = json.dumps({
        "model": "flash-moe",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "session_id": session_id,
        "stream": True,
    }).encode()

    url = f"http://{host}:{port}/v1/chat/completions"
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })

    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except urllib.error.URLError as e:
        if hasattr(e, 'reason') and isinstance(e.reason, ConnectionRefusedError):
            print(f"\n[error] Cannot connect to server on {host}:{port}.")
            print(f"        Start it: ./bin/infer --serve {port}")
        else:
            print(f"\n[error] {e}")
        return ""

    full_text = []
    t0 = time.monotonic()
    first_token = True
    in_think = False
    in_code_block = False

    buf = b""
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line or line == b"data: [DONE]":
                continue
            if not line.startswith(b"data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            token = delta.get("content", "")
            if not token:
                # Check for finish_reason
                if choices[0].get("finish_reason"):
                    break
                continue

            token = clean_text(token)

            if first_token:
                first_token = False
                t_first = time.monotonic() - t0

            full_text.append(token)
            print(token, end="", flush=True)

    elapsed = time.monotonic() - t0
    n_tok = len(full_text)
    tok_s = n_tok / elapsed if elapsed > 0 else 0
    ttft = t_first if not first_token else 0

    result = "".join(full_text)
    print(f"\n\n[{n_tok} tokens, {tok_s:.1f} tok/s, TTFT {ttft:.2f}s]\n")
    return result


def save_turn(session_id: str, role: str, content: str):
    """Append a turn to the session file."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = os.path.join(SESSIONS_DIR, f"{session_id}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"role": role, "content": content}) + "\n")


def new_session_id() -> str:
    return f"chat-{os.getpid()}-{int(time.monotonic() * 1e6)}"


def main():
    parser = argparse.ArgumentParser(description="Chat client for Flash-MoE")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Server host (default: localhost)")
    parser.add_argument("--tokens", "-t", type=int, default=512,
                        help="Max tokens per response (default: 512)")
    parser.add_argument("--resume", "-r", type=str, default=None,
                        help="Resume a session by ID")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List saved sessions and exit")
    parser.add_argument("--show-think", action="store_true",
                        help="Show <think> blocks in output")
    args = parser.parse_args()

    # List sessions
    if args.list_sessions:
        sessions = load_sessions()
        if sessions:
            print("Saved sessions:")
            for sid in sorted(sessions):
                size = os.path.getsize(sessions[sid])
                print(f"  {sid}  ({size:,} bytes)")
        else:
            print("No saved sessions.")
        return

    # Resume or create session
    if args.resume:
        session_id = args.resume
        path = os.path.join(SESSIONS_DIR, f"{session_id}.jsonl")
        if not os.path.isfile(path):
            print(f"No session: {session_id}")
            return
        print(f"Resumed session: {session_id}")
    else:
        session_id = new_session_id()
        print(f"Session: {session_id}")

    # Show status
    print(f"Server:  http://{args.host}:{args.port}")
    print()

    try:
        readline.parse_and_bind("set editing-mode emacs")
    except Exception:
        pass

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input[1:].strip()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd in ("clear", "c"):
                os.system("clear" if sys.platform != "win32" else "cls")
                continue
            elif cmd.startswith("sessions"):
                sessions = load_sessions()
                for sid in sorted(sessions):
                    size = os.path.getsize(sessions[sid])
                    print(f"  {sid}  ({size:,} bytes)")
                continue
            else:
                print(f"Unknown command: {cmd}")
                print("Commands: /quit /clear /sessions")
                continue

        save_turn(session_id, "user", user_input)
        response = stream_chat(
            args.host, args.port, user_input, args.tokens,
            session_id, args.show_think,
        )
        if response:
            save_turn(session_id, "assistant", response)


if __name__ == "__main__":
    main()
