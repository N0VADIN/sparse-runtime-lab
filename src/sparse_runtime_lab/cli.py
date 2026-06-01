"""Command-line interface for Sparse Runtime Lab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .analyzer import analyze_model
from .report import render_markdown_report
from .runner import build_command, run_smoke_test

DEFAULT_PROMPT = "Explain sparse inference in one paragraph."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PowerInfer-first model tester and compatibility reporter.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run deterministic model intake analysis.")
    analyze.add_argument("--model", required=True, help="Path to a GGUF or .powerinfer.gguf artifact.")
    analyze.add_argument("--output", help="Optional Markdown report path.")

    test = subparsers.add_parser("test", help="Run a PowerInfer/llama.cpp runtime smoke test.")
    test.add_argument("--runtime", required=True, help="Path to PowerInfer or llama.cpp binary.")
    test.add_argument("--model", required=True, help="Path to a GGUF or .powerinfer.gguf artifact.")
    test.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for the smoke test.")
    test.add_argument("--tokens", type=int, default=128, help="Number of tokens to request.")
    test.add_argument("--threads", type=int, default=8, help="Runtime thread count.")
    test.add_argument("--vram-budget", type=int, help="Optional PowerInfer VRAM budget.")
    test.add_argument("--timeout", type=int, default=120, help="Timeout in seconds.")
    test.add_argument("--output", help="Optional Markdown report path.")
    test.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the runtime.")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        analysis = analyze_model(args.model)
        report = render_markdown_report(analysis)
        _emit(report, args.output)
        return 0 if analysis.compatibility.value != "red" else 2

    if args.command == "test":
        analysis = analyze_model(args.model)
        extra_args = tuple(arg for arg in args.extra_args if arg != "--")
        command = build_command(
            args.runtime,
            args.model,
            args.prompt,
            args.tokens,
            args.threads,
            args.vram_budget,
            extra_args,
        )
        runtime = run_smoke_test(command, timeout_seconds=args.timeout)
        report = render_markdown_report(analysis, runtime)
        _emit(report, args.output)
        return 0 if runtime.passed and analysis.compatibility.value != "red" else 2

    return 2


def _emit(report: str, output: str | None) -> None:
    if output:
        Path(output).write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
