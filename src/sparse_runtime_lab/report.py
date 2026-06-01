"""Markdown rendering for Sparse Runtime Lab JSON reports."""

from __future__ import annotations

from typing import Any

from .models import Gate, LayoutCheck, ModelAnalysis, RuntimeResult
from .schema import artifact_report, dumps_report, final_artifact_gate, layout_report


def final_gate(analysis: ModelAnalysis, runtime: RuntimeResult | None) -> Gate:
    """Backward-compatible wrapper for artifact gate computation."""

    return final_artifact_gate(analysis, runtime)


def render_json_report(analysis: ModelAnalysis, runtime: RuntimeResult | None = None) -> str:
    """Render an artifact report as stable, machine-readable JSON."""

    return dumps_report(artifact_report(analysis, runtime))


def render_layout_json(check: LayoutCheck) -> str:
    """Render a PowerInfer layout check as JSON."""

    return dumps_report(layout_report(check))


def render_markdown_report(analysis: ModelAnalysis, runtime: RuntimeResult | None = None) -> str:
    """Render an artifact report as Markdown via the JSON schema."""

    return render_markdown_from_report(artifact_report(analysis, runtime))


def render_layout_markdown(check: LayoutCheck) -> str:
    """Render a PowerInfer layout check as Markdown via the JSON schema."""

    return render_markdown_from_report(layout_report(check))


def render_markdown_from_report(report: dict[str, Any]) -> str:
    """Render an existing JSON report dictionary as Markdown."""

    report_type = report.get("report_type")
    if report_type == "artifact":
        return _render_artifact_markdown(report)
    if report_type == "layout":
        return _render_layout_markdown(report)
    raise ValueError(f"unsupported report_type: {report_type!r}")


def _render_artifact_markdown(report: dict[str, Any]) -> str:
    result = report["result"]
    analysis = report["static_analysis"]
    runtime = report.get("runtime")
    planned_command = report.get("planned_command")

    lines = [
        "# Sparse Runtime Lab Report",
        "",
        f"**Result:** {result['emoji']} {result['label']}",
        "",
        "## Static artifact analysis",
        "",
        f"- Path: `{analysis['model_path']}`",
        f"- Format: `{analysis['format']}`",
        f"- Family: `{analysis['family']}`",
        f"- Activation: `{analysis['activation']}`",
        f"- Tokenizer: `{analysis['tokenizer']}`",
        f"- Quantization: `{analysis['quantization']}`",
        f"- LoRA/adapter hint: `{'yes' if analysis['has_lora'] else 'no'}`",
        f"- PowerInfer artifact hint: `{'yes' if analysis['is_powerinfer_artifact'] else 'no'}`",
        f"- Static gate: `{analysis['compatibility']['value']}`",
        "",
        "## Gate reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in analysis.get("reasons") or ("No deterministic warnings detected.",))

    if planned_command is not None:
        lines.extend(["", "## Planned smoke command", "", f"- Command: `{' '.join(planned_command)}`"])

    if runtime is not None:
        lines.extend(
            [
                "",
                "## Runtime smoke test evidence",
                "",
                f"- Command: `{' '.join(runtime['command'])}`",
                f"- Loaded: `{'yes' if runtime['loaded'] else 'no'}`",
                f"- First token/output: `{'yes' if runtime['first_token'] else 'no'}`",
                f"- Tokens/s: `{_fmt(runtime['tokens_per_second'])}`",
                f"- Peak memory MB: `{_fmt(runtime['peak_memory_mb'])}`",
                f"- Return code: `{runtime['return_code']}`",
                f"- Timed out: `{'yes' if runtime['timed_out'] else 'no'}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Runtime smoke test evidence",
                "",
                "- Not run. Static analysis only marks candidates; it does not prove PowerInfer readiness.",
            ]
        )

    lines.extend(
        [
            "",
            "## Next gates",
            "",
            "- Compare against a dense GGUF baseline before trusting quality deltas.",
            "- Add perplexity/KL/top-k agreement checks before approving converted SwiGLU models.",
            "- Verify chat template and EOS/BOS handling in both dense and sparse runtimes.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_layout_markdown(report: dict[str, Any]) -> str:
    result = report["result"]
    layout = report["layout"]
    lines = [
        "# PowerInfer Layout Check",
        "",
        f"**Result:** {result['emoji']} {result['label']}",
        "",
        f"- Root: `{layout['root']}`",
        f"- Executable: `{layout['executable'] or 'n/a'}`",
        "",
        "## Found paths",
        "",
    ]
    lines.extend(f"- `{path}`" for path in layout.get("found_paths") or ("n/a",))
    lines.extend(["", "## Missing paths", ""])
    lines.extend(f"- `{path}`" for path in layout.get("missing_paths") or ("none",))
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- {reason}" for reason in layout.get("reasons", ()))
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
