"""Command-line interface for Sparse Runtime Lab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .analyzer import analyze_model
from .layout import check_powerinfer_layout
from .report import render_json_report, render_layout_json, render_layout_markdown, render_markdown_report
from .runner import build_command, run_smoke_test

DEFAULT_PROMPT = "Explain sparse inference in one paragraph."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PowerInfer-first model tester and compatibility reporter.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run deterministic static artifact analysis.")
    analyze.add_argument("--model", required=True, help="Path to a GGUF or .powerinfer.gguf artifact.")
    _add_report_args(analyze)

    layout = subparsers.add_parser("check-layout", help="Check a local PowerInfer checkout/build layout without executing it.")
    layout.add_argument("--powerinfer-dir", required=True, help="Path to a local PowerInfer checkout or build directory.")
    _add_report_args(layout)

    test = subparsers.add_parser("test", help="Run a PowerInfer/llama.cpp runtime smoke test.")
    test.add_argument("--runtime", required=True, help="Path to PowerInfer or llama.cpp binary.")
    test.add_argument("--model", required=True, help="Path to a GGUF or .powerinfer.gguf artifact.")
    test.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for the smoke test.")
    test.add_argument("--tokens", type=int, default=128, help="Number of tokens to request.")
    test.add_argument("--threads", type=int, default=8, help="Runtime thread count.")
    test.add_argument("--vram-budget", type=int, help="Optional PowerInfer VRAM budget.")
    test.add_argument("--timeout", type=int, default=120, help="Timeout in seconds.")
    _add_report_args(test)
    test.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the runtime.")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        analysis = analyze_model(args.model)
        report = _render_artifact_report(args.report_format, analysis)
        _emit(report, args.output)
        return 0 if analysis.compatibility.value != "red" else 2

    if args.command == "check-layout":
        check = check_powerinfer_layout(args.powerinfer_dir)
        report = render_layout_json(check) if args.report_format == "json" else render_layout_markdown(check)
        _emit(report, args.output)
        return 0 if check.passed else 2

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
        report = _render_artifact_report(args.report_format, analysis, runtime)
        _emit(report, args.output)
        return 0 if runtime.passed and analysis.compatibility.value != "red" else 2

    return 2


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", dest="report_format", choices=("markdown", "json"), default="markdown", help="Report format.")
    parser.add_argument("--output", help="Optional report path. Defaults to stdout.")


def _render_artifact_report(report_format: str, *args: object) -> str:
    if report_format == "json":
        return render_json_report(*args)  # type: ignore[arg-type]
    return render_markdown_report(*args)  # type: ignore[arg-type]


def _emit(report: str, output: str | None) -> None:
    if output:
        Path(output).write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
