"""Runtime smoke-test execution and log parsing."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from .models import RuntimeResult

TOKENS_PER_SECOND_PATTERNS = (
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tok/s|tokens/s|tokens per second)", re.I),
    re.compile(r"eval time.*?([0-9]+(?:\.[0-9]+)?)\s*tokens per second", re.I | re.S),
)
MEMORY_PATTERNS = (
    re.compile(r"(?:peak|total)?\s*(?:ram|memory|mem).*?([0-9]+(?:\.[0-9]+)?)\s*(gb|mb)", re.I),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(gb|mb)\s*(?:ram|memory|mem|vram)", re.I),
)
LOAD_PATTERNS = ("loaded", "llm_load", "model loaded", "tensor")
TOKEN_PATTERNS = ("tok/s", "tokens/s", "tokens per second", "sampling", "eval time")


def build_command(
    runtime: str | Path,
    model: str | Path,
    prompt: str,
    tokens: int,
    threads: int,
    vram_budget: int | None = None,
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build a PowerInfer/llama.cpp-compatible smoke-test command."""

    command = [str(runtime), "-m", str(model), "-p", prompt, "-n", str(tokens), "-t", str(threads)]
    if vram_budget is not None:
        command.extend(["--vram-budget", str(vram_budget)])
    command.extend(extra_args)
    return tuple(command)


def run_smoke_test(command: Sequence[str], timeout_seconds: int = 120) -> RuntimeResult:
    """Run a runtime binary and parse coarse readiness metrics."""

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return_code = 124
        timed_out = True

    combined = f"{stdout}\n{stderr}"
    elapsed = max(time.monotonic() - started, 0.001)
    tokens_per_second = _parse_tokens_per_second(combined)
    if tokens_per_second is None and return_code == 0:
        tokens_per_second = None

    return RuntimeResult(
        command=tuple(command),
        loaded=_contains_any(combined, LOAD_PATTERNS),
        first_token=_contains_any(combined, TOKEN_PATTERNS) or (return_code == 0 and len(stdout.strip()) > 0 and elapsed > 0),
        tokens_per_second=tokens_per_second,
        peak_memory_mb=_parse_memory_mb(combined),
        return_code=return_code,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
        timed_out=timed_out,
    )


def parse_runtime_output(command: Sequence[str], stdout: str, stderr: str = "", return_code: int = 0) -> RuntimeResult:
    """Parse captured runtime output; useful for fixtures and imported reports."""

    combined = f"{stdout}\n{stderr}"
    return RuntimeResult(
        command=tuple(command),
        loaded=_contains_any(combined, LOAD_PATTERNS),
        first_token=_contains_any(combined, TOKEN_PATTERNS) or (return_code == 0 and bool(stdout.strip())),
        tokens_per_second=_parse_tokens_per_second(combined),
        peak_memory_mb=_parse_memory_mb(combined),
        return_code=return_code,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


def _parse_tokens_per_second(text: str) -> float | None:
    for pattern in TOKENS_PER_SECOND_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _parse_memory_mb(text: str) -> float | None:
    for pattern in MEMORY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value * 1024 if unit == "gb" else value
    return None


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])
