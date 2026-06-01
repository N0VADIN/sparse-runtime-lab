"""Schema objects for activation profiling plans and reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CalibrationSource:
    """Local calibration prompt source for a future profiler run."""

    path: Path
    exists: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProfilingPlan:
    """Dry-run activation profiling plan; it does not load models."""

    model_path: Path
    calibration: CalibrationSource
    max_samples: int
    target_modules: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LayerSparsitySummary:
    """Per-layer activation sparsity summary placeholder for MVP 2 reports."""

    layer_index: int
    module_name: str
    total_values: int = 0
    zero_values: int = 0
    sparsity: float | None = None


@dataclass(frozen=True)
class ActivationProfileReport:
    """Activation profiling report skeleton for future measured summaries."""

    plan: ProfilingPlan
    layers: tuple[LayerSparsitySummary, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def profiling_plan_report(plan: ProfilingPlan) -> dict[str, Any]:
    """Build a JSON-compatible dry-run profiling plan report."""

    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "profile_plan",
        "profile_plan": profiling_plan_to_dict(plan),
    }


def activation_profile_report(report: ActivationProfileReport) -> dict[str, Any]:
    """Build a JSON-compatible activation profile report skeleton."""

    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "activation_profile",
        "profile_plan": profiling_plan_to_dict(report.plan),
        "layers": [layer_sparsity_summary_to_dict(layer) for layer in report.layers],
        "warnings": list(report.warnings),
    }


def profiling_plan_to_dict(plan: ProfilingPlan) -> dict[str, Any]:
    """Serialize a profiling plan."""

    return {
        "model_path": str(plan.model_path),
        "calibration": calibration_source_to_dict(plan.calibration),
        "max_samples": plan.max_samples,
        "target_modules": list(plan.target_modules),
        "warnings": list(plan.warnings),
    }


def calibration_source_to_dict(source: CalibrationSource) -> dict[str, Any]:
    """Serialize a calibration source."""

    return {
        "path": str(source.path),
        "exists": source.exists,
        "warnings": list(source.warnings),
    }


def layer_sparsity_summary_to_dict(summary: LayerSparsitySummary) -> dict[str, Any]:
    """Serialize a layer sparsity summary."""

    return {
        "layer_index": summary.layer_index,
        "module_name": summary.module_name,
        "total_values": summary.total_values,
        "zero_values": summary.zero_values,
        "sparsity": summary.sparsity,
    }


def dumps_profile_report(report: dict[str, Any]) -> str:
    """Dump a profiling report dictionary as stable JSON."""

    return json.dumps(report, indent=2, sort_keys=True) + "\n"
