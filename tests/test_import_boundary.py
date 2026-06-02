from __future__ import annotations

import ast
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "src" / "sparse_runtime_lab" / "runner.py"


def test_runner_stays_on_runtime_boundary():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.endswith(".analyzer"), alias.name
                assert not alias.name.endswith(".profiling"), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_names = {alias.name for alias in node.names}

            assert not module.endswith("analyzer"), module
            assert not module.endswith("profiling"), module

            if module in {"", "sparse_runtime_lab"}:
                assert "analyzer" not in imported_names, imported_names
                assert "profiling" not in imported_names, imported_names

            if module.endswith("models"):
                assert imported_names <= {"RuntimeResult"}, imported_names
