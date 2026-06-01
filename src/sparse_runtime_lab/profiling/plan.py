"""Dry-run activation profiling plan construction."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from .schema import CalibrationSource, ProfilingPlan


def create_profiling_plan(
    model_path: str | Path,
    calibration_path: str | Path,
    max_samples: int,
    target_modules: Iterable[str],
) -> ProfilingPlan:
    """Create a profiling plan without loading models or calibration data."""

    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")

    calibration = _calibration_source(Path(calibration_path))
    modules = tuple(module for module in target_modules if module)
    warnings: list[str] = []
    if not modules:
        warnings.append("No target modules were provided; profiling would have no hook targets.")

    return ProfilingPlan(
        model_path=Path(model_path),
        calibration=calibration,
        max_samples=max_samples,
        target_modules=modules,
        warnings=tuple(warnings),
    )


def _calibration_source(path: Path) -> CalibrationSource:
    warnings: list[str] = []
    exists = path.exists()
    if not exists:
        warnings.append("Calibration file does not exist; this is a dry-run plan only.")
    return CalibrationSource(path=path, exists=exists, warnings=tuple(warnings))
