"""PowerInfer checkout/build layout checks."""

from __future__ import annotations

import os
from pathlib import Path

from .models import Gate, LayoutCheck

EXECUTABLE_CANDIDATES = (
    "build/bin/main",
    "build/bin/powerinfer",
    "main",
    "powerinfer",
)
SOURCE_HINTS = (
    "CMakeLists.txt",
    "README.md",
)


def check_powerinfer_layout(root: str | Path) -> LayoutCheck:
    """Check whether a PowerInfer-like directory has a runnable binary.

    The check is intentionally conservative and filesystem-only. It does not
    build PowerInfer, download models, or execute any binary.
    """

    root_path = Path(root)
    found: list[Path] = []
    missing: list[str] = []
    reasons: list[str] = []

    if not root_path.exists():
        return LayoutCheck(
            root=root_path,
            executable=None,
            found_paths=(),
            missing_paths=("directory",),
            gate=Gate.RED,
            reasons=("PowerInfer directory does not exist.",),
        )
    if not root_path.is_dir():
        return LayoutCheck(
            root=root_path,
            executable=None,
            found_paths=(),
            missing_paths=("directory",),
            gate=Gate.RED,
            reasons=("PowerInfer path is not a directory.",),
        )

    executable = _find_executable(root_path)
    if executable is not None:
        found.append(executable)
        reasons.append("Runnable PowerInfer-style binary found.")
    else:
        missing.append("build/bin/main or equivalent executable")
        reasons.append("No executable runtime binary found; build PowerInfer before running smoke tests.")

    for hint in SOURCE_HINTS:
        candidate = root_path / hint
        if candidate.exists():
            found.append(candidate)
        else:
            missing.append(hint)

    if executable is None:
        gate = Gate.RED
    elif missing:
        gate = Gate.YELLOW
        reasons.append("Runtime exists, but some source-layout hints are missing; verify this is the intended checkout.")
    else:
        gate = Gate.YELLOW
        reasons.append("Layout is runnable; run a model smoke test for PowerInfer-ready evidence.")

    return LayoutCheck(
        root=root_path,
        executable=executable,
        found_paths=tuple(found),
        missing_paths=tuple(missing),
        gate=gate,
        reasons=tuple(reasons),
    )


def _find_executable(root: Path) -> Path | None:
    for relative in EXECUTABLE_CANDIDATES:
        candidate = root / relative
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None
