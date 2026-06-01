"""Data models for Sparse Runtime Lab reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Gate(str, Enum):
    """Traffic-light gate status for model/runtime readiness."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

    @property
    def emoji(self) -> str:
        return {
            Gate.GREEN: "🟢",
            Gate.YELLOW: "🟡",
            Gate.RED: "🔴",
        }[self]

    @property
    def label(self) -> str:
        return {
            Gate.GREEN: "PowerInfer-ready",
            Gate.YELLOW: "Experimental",
            Gate.RED: "Not suitable",
        }[self]


@dataclass(frozen=True)
class ModelAnalysis:
    """Deterministic model intake analysis based on file naming and metadata."""

    model_path: Path
    format: str
    family: str
    activation: str
    tokenizer: str
    quantization: str
    has_lora: bool
    is_powerinfer_artifact: bool
    compatibility: Gate
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RuntimeResult:
    """Captured outcome of a dense or sparse runtime smoke test."""

    command: tuple[str, ...]
    loaded: bool
    first_token: bool
    tokens_per_second: float | None
    peak_memory_mb: float | None
    return_code: int
    stdout_tail: str
    stderr_tail: str
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.return_code == 0 and self.loaded and self.first_token and not self.timed_out
