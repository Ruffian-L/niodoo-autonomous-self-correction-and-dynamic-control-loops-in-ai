#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DECODE_RE = re.compile(r"\[DBG: Decoded '(.*)'\]")
TELEMETRY_PREFIX = "[TELEMETRY]"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_binary(root: Path) -> Path:
    release = root / "target" / "release" / "niodoo"
    debug = root / "target" / "debug" / "niodoo"
    if release.exists():
        return release
    if debug.exists():
        return debug
    return release


def decode_fragment(line: str) -> str | None:
    match = DECODE_RE.search(line)
    if not match:
        return None
    return match.group(1).replace("\\n", "\n").replace("\\'", "'")


def parse_telemetry(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped.startswith(TELEMETRY_PREFIX):
        return None
    payload = stripped[len(TELEMETRY_PREFIX) :].strip().rstrip(",")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def ensure_telemetry_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()


def build_prompt(history: list[tuple[str, str]], user_message: str) -> str:
    if not history:
        return user_message

    transcript = []
    for role, content in history:
        transcript.append(f"{role}: {content}")
    transcript.append(f"user: {user_message}")
    transcript.append("assistant:")
    return (
        "Continue this conversation exactly as an ongoing local chat. Preserve raw Niodoo behavior, including emitted control tags and telemetry.\n\n"
        + "\n".join(transcript)
    )


def run_turn(args: argparse.Namespace, prompt: str, telemetry_path: Path, turn_index: int) -> str:
    cmd = [
        str(args.binary),
        "--model-path",
        str(args.model),
        "--particles-path",
        str(args.particles),
        "--prompt",
        prompt,
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--physics-blend",
        str(args.physics_blend),
        f"--repulsion-strength={args.repulsion_strength}",
        "--gravity-well",
        str(args.gravity_well),
        "--orbit-speed",
        str(args.orbit_speed),
    ]
    if args.mode_orbital:
        cmd.append("--mode-orbital")
    if args.n is not None:
        cmd.extend(["--n", str(args.n)])
    for extra_arg in args.extra_arg:
        cmd.append(extra_arg)

    assistant_text: list[str] = []
    telemetry: list[dict] = []
    diagnostics: list[str] = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    print("assistant> ", end="", flush=True)
    for raw_line in process.stdout:
        fragment = decode_fragment(raw_line)
        if fragment is not None:
            assistant_text.append(fragment)
            sys.stdout.write(fragment)
            sys.stdout.flush()
            continue

        telemetry_frame = parse_telemetry(raw_line)
        if telemetry_frame is not None:
            telemetry.append(telemetry_frame)
            continue

        diagnostics.append(raw_line.rstrip("\n"))

    print()

    returncode = process.wait()
    record = {
        "turn": turn_index,
        "prompt": prompt,
        "assistant_text": "".join(assistant_text),
        "telemetry": telemetry,
        "diagnostics": diagnostics,
        "returncode": returncode,
    }
    with telemetry_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if returncode != 0:
        tail = "\n".join(diagnostics[-20:])
        raise RuntimeError(f"niodoo exited with status {returncode}\n{tail}")

    return "".join(assistant_text).strip()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Raw multi-turn CLI for the local Niodoo binary.")
    parser.add_argument("--binary", type=Path, default=default_binary(root))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            os.environ.get(
                "NIODOO_MODEL_PATH",
                root / "model" / "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
            )
        ),
    )
    parser.add_argument(
        "--particles",
        type=Path,
        default=Path(
            os.environ.get(
                "NIODOO_PARTICLES_PATH",
                root / "universe_top60000.safetensors",
            )
        ),
    )
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-blend", type=float, default=1.5)
    parser.add_argument("--repulsion-strength", type=float, default=-0.5)
    parser.add_argument("--gravity-well", type=float, default=0.2)
    parser.add_argument("--orbit-speed", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=60000)
    parser.add_argument("--mode-orbital", action="store_true", default=True)
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        default=root / "artifacts" / "chat_raw",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument to pass directly to the Rust binary. Repeat as needed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}", file=sys.stderr)
        print("Build it with: cargo build --release --bin niodoo", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1
    if not args.particles.exists():
        print(f"Universe not found: {args.particles}", file=sys.stderr)
        return 1

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    telemetry_path = args.telemetry_dir / f"chat_raw_{session_id}.jsonl"
    ensure_telemetry_file(telemetry_path)

    history: list[tuple[str, str]] = []
    turn_index = 0
    print("Niodoo chat. Commands: /reset, /history, /exit")
    print(f"telemetry: {telemetry_path}")

    while True:
        try:
            user_message = input("\nuser> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not user_message:
            continue
        if user_message == "/exit":
            break
        if user_message == "/reset":
            history.clear()
            print("history cleared")
            continue
        if user_message == "/history":
            if not history:
                print("history is empty")
            else:
                for role, content in history:
                    print(f"{role}: {content}")
            continue

        prompt = build_prompt(history, user_message)
        try:
            assistant = run_turn(args, prompt, telemetry_path, turn_index)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue

        history.append(("user", user_message))
        history.append(("assistant", assistant))
        turn_index += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
