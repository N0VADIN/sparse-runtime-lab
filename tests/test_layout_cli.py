import json
import os

from sparse_runtime_lab.cli import main
from sparse_runtime_lab.layout import check_powerinfer_layout
from sparse_runtime_lab.models import Gate
from sparse_runtime_lab.report import render_layout_json


def test_layout_missing_directory_is_red(tmp_path):
    check = check_powerinfer_layout(tmp_path / "missing")

    assert check.gate is Gate.RED
    assert check.executable is None
    assert "directory" in check.missing_paths


def test_layout_finds_executable_without_running_it(tmp_path):
    root = tmp_path / "PowerInfer"
    binary = root / "build" / "bin" / "main"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | os.X_OK)
    (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    (root / "README.md").write_text("# test\n", encoding="utf-8")

    check = check_powerinfer_layout(root)

    assert check.gate is Gate.YELLOW
    assert check.executable == binary
    assert check.missing_paths == ()
    assert "Runnable" in " ".join(check.reasons)


def test_layout_json_report(tmp_path):
    check = check_powerinfer_layout(tmp_path / "missing")
    payload = json.loads(render_layout_json(check))

    assert payload["result"]["value"] == "red"
    assert payload["layout"]["executable"] is None


def test_cli_analyze_can_emit_json(capsys):
    rc = main(["analyze", "--model", "SmallThinker-Q4_K_M.powerinfer.gguf", "--format", "json"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["result"]["value"] == "yellow"
    assert payload["runtime"] is None
