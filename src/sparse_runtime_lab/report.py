"""JSON and Markdown reporting for Sparse Runtime Lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Gate, LayoutCheck, ModelAnalysis, RuntimeResult


def final_gate(analysis: ModelAnalysis, runtime: RuntimeResult | None) -> Gate:
    """Combine static compatibility and optional runtime smoke-test status.

    Static analysis alone never upgrades a model to green. Green means the
    artifact is not statically blocked and there is passing runtime evidence.
    """

    if analysis.compatibility is Gate.RED:
        return Gate.RED
    if runtime is None:
        return analysis.compatibility
    if runtime.passed:
        return Gate.GREEN
    return Gate.RED


def render_markdown_report(analysis: ModelAnalysis, runtime: RuntimeResult | None = None) -> str:
    """Render a concise traffic-light report for an artifact."""

    gate = final_gate(analysis, runtime)
    lines = [
        "# Sparse Runtime Lab Report",
        "",
        f"**Result:** {gate.emoji} {gate.label}",
        "",
        "## Static artifact analysis",
        "",
        f"- Path: `{analysis.model_path}`",
        f"- Format: `{analysis.format}`",
        f"- Family: `{analysis.family}`",
        f"- Activation: `{analysis.activation}`",
        f"- Tokenizer: `{analysis.tokenizer}`",
        f"- Quantization: `{analysis.quantization}`",
        f"- LoRA/adapter hint: `{'yes' if analysis.has_lora else 'no'}`",
        f"- PowerInfer artifact hint: `{'yes' if analysis.is_powerinfer_artifact else 'no'}`",
        f"- Static gate: `{analysis.compatibility.value}`",
        "",
        "## Gate reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in analysis.reasons or ("No deterministic warnings detected.",))

    if runtime is not None:
        lines.extend(
            [
                "",
                "## Runtime smoke test evidence",
                "",
                f"- Command: `{' '.join(runtime.command)}`",
                f"- Loaded: `{'yes' if runtime.loaded else 'no'}`",
                f"- First token/output: `{'yes' if runtime.first_token else 'no'}`",
                f"- Tokens/s: `{_fmt(runtime.tokens_per_second)}`",
                f"- Peak memory MB: `{_fmt(runtime.peak_memory_mb)}`",
                f"- Return code: `{runtime.return_code}`",
                f"- Timed out: `{'yes' if runtime.timed_out else 'no'}`",
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


def render_json_report(analysis: ModelAnalysis, runtime: RuntimeResult | None = None) -> str:
    """Render an artifact report as stable, machine-readable JSON."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "result": _gate_to_dict(final_gate(analysis, runtime)),
        "static_analysis": _analysis_to_dict(analysis),
        "runtime": _runtime_to_dict(runtime) if runtime is not None else None,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_layout_markdown(check: LayoutCheck) -> str:
    """Render a PowerInfer layout check as Markdown."""

    lines = [
        "# PowerInfer Layout Check",
        "",
        f"**Result:** {check.gate.emoji} {check.gate.label}",
        "",
        f"- Root: `{check.root}`",
        f"- Executable: `{check.executable or 'n/a'}`",
        "",
        "## Found paths",
        "",
    ]
    lines.extend(f"- `{path}`" for path in check.found_paths or (Path("n/a"),))
    lines.extend(["", "## Missing paths", ""])
    lines.extend(f"- `{path}`" for path in check.missing_paths or ("none",))
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- {reason}" for reason in check.reasons)
    lines.append("")
    return "\n".join(lines)


def render_layout_json(check: LayoutCheck) -> str:
    """Render a PowerInfer layout check as JSON."""

    payload = {
        "schema_version": 1,
        "result": _gate_to_dict(check.gate),
        "layout": {
            "root": str(check.root),
            "executable": str(check.executable) if check.executable else None,
            "found_paths": [str(path) for path in check.found_paths],
            "missing_paths": list(check.missing_paths),
            "reasons": list(check.reasons),
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _analysis_to_dict(analysis: ModelAnalysis) -> dict[str, Any]:
    return {
        "model_path": str(analysis.model_path),
        "format": analysis.format,
        "family": analysis.family,
        "activation": analysis.activation,
        "tokenizer": analysis.tokenizer,
        "quantization": analysis.quantization,
        "has_lora": analysis.has_lora,
        "is_powerinfer_artifact": analysis.is_powerinfer_artifact,
        "compatibility": _gate_to_dict(analysis.compatibility),
        "reasons": list(analysis.reasons),
    }


def _runtime_to_dict(runtime: RuntimeResult) -> dict[str, Any]:
    return {
        "command": list(runtime.command),
        "loaded": runtime.loaded,
        "first_token": runtime.first_token,
        "tokens_per_second": runtime.tokens_per_second,
        "peak_memory_mb": runtime.peak_memory_mb,
        "return_code": runtime.return_code,
        "timed_out": runtime.timed_out,
        "passed": runtime.passed,
        "stdout_tail": runtime.stdout_tail,
        "stderr_tail": runtime.stderr_tail,
    }


def _gate_to_dict(gate: Gate) -> dict[str, str]:
    return {"value": gate.value, "emoji": gate.emoji, "label": gate.label}


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
