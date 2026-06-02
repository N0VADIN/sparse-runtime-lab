"""Command-line interface for Sparse Runtime Lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .analyzer import analyze_model
from .layout import check_powerinfer_layout
from .profiling import create_profiling_plan, dumps_profile_report, profiling_plan_report
from .report import render_markdown_from_report
from .runner import build_command, run_smoke_test
from .schema import artifact_report, dumps_report, layout_report, load_report

DEFAULT_PROMPT = "Explain sparse inference in one paragraph."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PowerInfer-first model tester and compatibility reporter.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        aliases=["export-metadata"],
        help="Run deterministic static artifact analysis.",
    )
    analyze.add_argument("--model", required=True, help="Path to a GGUF or .powerinfer.gguf artifact.")
    _add_output_arg(analyze)

    layout = subparsers.add_parser("check-layout", help="Check a local PowerInfer checkout/build layout without executing it.")
    layout.add_argument("--powerinfer-dir", required=True, help="Path to a local PowerInfer checkout or build directory.")
    _add_output_arg(layout)

    smoke = subparsers.add_parser("smoke", help="Build or run a PowerInfer/llama.cpp runtime smoke test.")
    _add_smoke_args(smoke)

    bench = subparsers.add_parser(
        "bench",
        help="Baseline benchmark namespace that currently wraps the dense smoke path.",
    )
    bench_subparsers = bench.add_subparsers(dest="bench_command", required=True)
    bench_dense = bench_subparsers.add_parser(
        "dense",
        help="Run the current dense/runtime smoke path as the baseline benchmark wrapper.",
    )
    _add_smoke_args(bench_dense)

    profile_plan = subparsers.add_parser(
        "profile-plan",
        aliases=["profile"],
        help="Create a dry-run activation profiling plan JSON.",
    )
    profile_plan.add_argument("--model", required=True, help="Path to the model artifact or model directory.")
    profile_plan.add_argument("--calibration", required=True, help="Path to a local calibration prompt file.")
    profile_plan.add_argument("--max-samples", type=int, required=True, help="Maximum calibration samples to plan for.")
    profile_plan.add_argument("--target-modules", nargs="+", required=True, help="Module names to target in a future profiler run.")
    _add_output_arg(profile_plan)

    report = subparsers.add_parser("report", help="Render an existing Sparse Runtime Lab JSON report to Markdown.")
    report.add_argument("--input", required=True, help="Path to an existing JSON report.")
    _add_output_arg(report)

    args = parser.parse_args(argv)

    if args.command in {"analyze", "export-metadata"}:
        analysis = analyze_model(args.model)
        _emit(dumps_report(artifact_report(analysis)), args.output)
        return 0 if analysis.compatibility.value != "red" else 2

    if args.command == "check-layout":
        check = check_powerinfer_layout(args.powerinfer_dir)
        _emit(dumps_report(layout_report(check)), args.output)
        return 0 if check.passed else 2

    if args.command == "smoke":
        return _run_smoke(args)

    if args.command == "bench":
        if args.bench_command == "dense":
            return _run_smoke(args)
        return 2

    if args.command in {"profile-plan", "profile"}:
        plan = create_profiling_plan(args.model, args.calibration, args.max_samples, args.target_modules)
        _emit(dumps_profile_report(profiling_plan_report(plan)), args.output)
        return 0

    if args.command == "report":
        try:
            report = load_report(args.input)
            markdown = render_markdown_from_report(report)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        _emit(markdown, args.output)
        return 0

    return 2


def _add_smoke_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime", required=True, help="Path to PowerInfer or llama.cpp binary.")
    parser.add_argument("--model", required=True, help="Path to a GGUF or .powerinfer.gguf artifact.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for the smoke test.")
    parser.add_argument("--tokens", type=int, default=128, help="Number of tokens to request.")
    parser.add_argument("--threads", type=int, default=8, help="Runtime thread count.")
    parser.add_argument("--vram-budget", type=int, help="Optional PowerInfer VRAM budget.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Only build and report the command; do not execute the runtime.")
    _add_output_arg(parser)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the runtime.")


def _run_smoke(args: argparse.Namespace) -> int:
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
    if args.dry_run:
        _emit(dumps_report(artifact_report(analysis, planned_command=command)), args.output)
        return 0 if analysis.compatibility.value != "red" else 2
    runtime = run_smoke_test(command, timeout_seconds=args.timeout)
    _emit(dumps_report(artifact_report(analysis, runtime)), args.output)
    return 0 if runtime.passed and analysis.compatibility.value != "red" else 2


def _add_output_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="Optional output path. Defaults to stdout.")


def _emit(content: str, output: str | None) -> None:
    if output:
        Path(output).write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
