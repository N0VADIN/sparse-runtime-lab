"""Machine-readable JSON report schema helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Gate, LayoutCheck, ModelAnalysis, RuntimeResult

SCHEMA_VERSION = 1


def gate_to_dict(gate: Gate) -> dict[str, str]:
    """Serialize a traffic-light gate."""

    return {"value": gate.value, "emoji": gate.emoji, "label": gate.label}


def final_artifact_gate(analysis: ModelAnalysis, runtime: RuntimeResult | None) -> Gate:
    """Compute final artifact readiness.

    Static analysis never upgrades a report to green; green requires a non-red
    static candidate plus passing runtime evidence.
    """

    if analysis.compatibility is Gate.RED:
        return Gate.RED
    if runtime is None:
        return analysis.compatibility
    if not runtime.passed:
        return Gate.RED
    if analysis.is_powerinfer_artifact:
        return Gate.GREEN
    return analysis.compatibility


def artifact_report(
    analysis: ModelAnalysis,
    runtime: RuntimeResult | None = None,
    planned_command: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a versioned artifact report dictionary."""

    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "artifact",
        "result": gate_to_dict(final_artifact_gate(analysis, runtime)),
        "static_analysis": analysis_to_dict(analysis),
        "planned_command": list(planned_command) if planned_command is not None else None,
        "runtime": runtime_to_dict(runtime) if runtime is not None else None,
    }


def layout_report(check: LayoutCheck) -> dict[str, Any]:
    """Build a versioned PowerInfer layout report dictionary."""

    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "layout",
        "result": gate_to_dict(check.gate),
        "layout": layout_to_dict(check),
    }


def analysis_to_dict(analysis: ModelAnalysis) -> dict[str, Any]:
    """Serialize static artifact analysis."""

    return {
        "model_path": str(analysis.model_path),
        "format": analysis.format,
        "family": analysis.family,
        "activation": analysis.activation,
        "tokenizer": analysis.tokenizer,
        "quantization": analysis.quantization,
        "has_lora": analysis.has_lora,
        "is_powerinfer_artifact": analysis.is_powerinfer_artifact,
        "compatibility": gate_to_dict(analysis.compatibility),
        "reasons": list(analysis.reasons),
    }


def runtime_to_dict(runtime: RuntimeResult) -> dict[str, Any]:
    """Serialize runtime smoke-test evidence."""

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


def layout_to_dict(check: LayoutCheck) -> dict[str, Any]:
    """Serialize a PowerInfer layout check."""

    return {
        "root": str(check.root),
        "executable": str(check.executable) if check.executable else None,
        "found_paths": [str(path) for path in check.found_paths],
        "missing_paths": list(check.missing_paths),
        "reasons": list(check.reasons),
    }


def dumps_report(report: dict[str, Any]) -> str:
    """Dump a report dictionary as stable JSON."""

    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a report JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("report JSON must contain an object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported report schema version: {data.get('schema_version')!r}")
    return data
