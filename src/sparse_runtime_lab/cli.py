"""Command line interface for Sparse Runtime Lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sparse_runtime_lab.gguf import GgufInspectionError, inspect_gguf_header
from sparse_runtime_lab.powerinfer import (
    build_powerinfer_smoke_command,
    inspect_powerinfer_dir,
    shell_quote_command,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = args.func(args)
    except (FileNotFoundError, NotADirectoryError, GgufInspectionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if result is not None:
        if getattr(args, "as_json", False):
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            _print_human(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="srl", description="Sparse Runtime Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gguf_parser = subparsers.add_parser("inspect-gguf", help="Inspect a minimal GGUF header")
    gguf_parser.add_argument("path", type=Path)
    gguf_parser.add_argument("--json", action="store_true", dest="as_json")
    gguf_parser.set_defaults(func=_cmd_inspect_gguf)

    pi_parser = subparsers.add_parser(
        "inspect-powerinfer-dir", help="Inspect a PowerInfer-style model directory"
    )
    pi_parser.add_argument("path", type=Path)
    pi_parser.add_argument("--json", action="store_true", dest="as_json")
    pi_parser.set_defaults(func=_cmd_inspect_powerinfer_dir)

    smoke_parser = subparsers.add_parser("run-smoke", help="Build a PowerInfer smoke-test command")
    smoke_parser.add_argument("--binary", required=True, type=Path)
    smoke_parser.add_argument("--model", required=True, type=Path)
    smoke_parser.add_argument("--prompt", required=True)
    smoke_parser.add_argument("--tokens", type=int, default=128)
    smoke_parser.add_argument("--threads", type=int, default=8)
    smoke_parser.add_argument("--vram-budget", type=float, default=None)
    smoke_parser.add_argument("--extra-arg", action="append", default=[])
    smoke_parser.add_argument("--dry-run", action="store_true", default=False)
    smoke_parser.add_argument("--json", action="store_true", dest="as_json")
    smoke_parser.set_defaults(func=_cmd_run_smoke)

    return parser


def _cmd_inspect_gguf(args: argparse.Namespace) -> dict[str, object]:
    return inspect_gguf_header(args.path).to_dict()


def _cmd_inspect_powerinfer_dir(args: argparse.Namespace) -> dict[str, object]:
    return inspect_powerinfer_dir(args.path).to_dict()


def _cmd_run_smoke(args: argparse.Namespace) -> dict[str, object]:
    command = build_powerinfer_smoke_command(
        binary=args.binary,
        model=args.model,
        prompt=args.prompt,
        tokens=args.tokens,
        threads=args.threads,
        vram_budget=args.vram_budget,
        extra_args=args.extra_arg,
    )
    return {
        "dry_run": bool(args.dry_run),
        "command": command,
        "shell": shell_quote_command(command),
        "note": "Execution is intentionally not implemented yet; this command builder is safe to test.",
    }


def _print_human(result: dict[str, object]) -> None:
    for key, value in result.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
