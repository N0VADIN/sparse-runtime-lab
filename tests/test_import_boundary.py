from __future__ import annotations

import ast
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "src" / "sparse_runtime_lab" / "runner.py"
BLOCKED_IMPORT_SEGMENTS = {"analyzer", "profiling"}


def _is_blocked_module(name: str) -> bool:
    return any(segment in BLOCKED_IMPORT_SEGMENTS for segment in name.split("."))


def test_runner_stays_on_runtime_boundary():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not _is_blocked_module(alias.name), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_names = {alias.name for alias in node.names}

            assert not _is_blocked_module(module), module

            if module in {"", "sparse_runtime_lab"}:
                assert "analyzer" not in imported_names, imported_names
                assert "profiling" not in imported_names, imported_names

            if module.endswith("models"):
                assert imported_names <= {"RuntimeResult"}, imported_names
