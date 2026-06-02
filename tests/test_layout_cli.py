import json
from pathlib import Path

import pytest
from sparse_runtime_lab.cli import main
from sparse_runtime_lab.layout import check_powerinfer_layout
from sparse_runtime_lab.models import Gate
from sparse_runtime_lab.report import render_markdown_from_report
from sparse_runtime_lab.schema import layout_report


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
    binary.chmod(0o755)
    (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    (root / "README.md").write_text("# test\n", encoding="utf-8")

    check = check_powerinfer_layout(root)

    assert check.gate is Gate.YELLOW
    assert check.executable == binary
    assert check.missing_paths == ()
    assert "Runnable" in " ".join(check.reasons)


def test_layout_schema_report(tmp_path):
    check = check_powerinfer_layout(tmp_path / "missing")
    payload = layout_report(check)
    markdown = render_markdown_from_report(payload)

    assert payload["result"]["value"] == "red"
    assert payload["layout"]["executable"] is None
    assert "# PowerInfer Layout Check" in markdown


def test_cli_analyze_emits_json(capsys):
    rc = main(["analyze", "--model", "SmallThinker-Q4_K_M.powerinfer.gguf"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["report_type"] == "artifact"
    assert payload["result"]["value"] == "yellow"
    assert payload["runtime"] is None


def test_cli_smoke_dry_run_builds_command_without_executing(capsys):
    rc = main(["smoke", "--runtime", "/missing/runtime", "--model", "SmallThinker-Q4_K_M.powerinfer.gguf", "--dry-run"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["report_type"] == "artifact"
    assert payload["runtime"] is None
    assert payload["planned_command"][0] == "/missing/runtime"


def test_cli_report_renders_existing_json(tmp_path, capsys):
    json_path = tmp_path / "artifact.json"
    rc = main(["analyze", "--model", "SmallThinker-Q4_K_M.powerinfer.gguf", "--output", str(json_path)])
    assert rc == 0

    rc = main(["report", "--input", str(json_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "# Sparse Runtime Lab Report" in out


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        (None, "No such file or directory"),
        ("not json", "Expecting value"),
        ('{"schema_version": 1, "report_type": "unknown"}', "unsupported report type"),
        ('{"schema_version": 1, "report_type": "artifact", "result": {}}', "missing required top-level key"),
    ],
)
def test_cli_report_returns_clean_error_for_expected_failures(tmp_path, capsys, payload, expected_error):
    report_path = tmp_path / "bad.json"
    if payload is not None:
        report_path.write_text(payload, encoding="utf-8")

    rc = main(["report", "--input", str(report_path)])
    captured = capsys.readouterr()

    assert rc == 2
    assert expected_error in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_report_returns_clean_error_for_renderer_value_error(tmp_path, capsys, monkeypatch):
    report_path = tmp_path / "bad.json"
    report_path.write_text(
        '{"schema_version": 1, "report_type": "artifact", "result": {}, "static_analysis": {}}',
        encoding="utf-8",
    )

    def fake_renderer(_report):
        raise ValueError("renderer exploded")

    monkeypatch.setattr("sparse_runtime_lab.cli.render_markdown_from_report", fake_renderer)

    rc = main(["report", "--input", str(report_path)])
    captured = capsys.readouterr()

    assert rc == 2
    assert "renderer exploded" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_pyproject_exposes_both_console_scripts():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'sparse-runtime-lab = "sparse_runtime_lab.cli:main"' in pyproject
    assert 'srl = "sparse_runtime_lab.cli:main"' in pyproject


def test_cli_export_metadata_alias_matches_analyze(capsys):
    rc = main(["analyze", "--model", "SmallThinker-Q4_K_M.powerinfer.gguf"])
    analyze_out = capsys.readouterr().out

    rc_alias = main(["export-metadata", "--model", "SmallThinker-Q4_K_M.powerinfer.gguf"])
    alias_out = capsys.readouterr().out

    assert rc == 0
    assert rc_alias == 0
    assert analyze_out == alias_out


def test_cli_bench_dense_routes_through_smoke_behavior(capsys):
    smoke_rc = main(["smoke", "--runtime", "/missing/runtime", "--model", "SmallThinker-Q4_K_M.powerinfer.gguf", "--dry-run"])
    smoke_out = capsys.readouterr().out

    bench_rc = main(["bench", "dense", "--runtime", "/missing/runtime", "--model", "SmallThinker-Q4_K_M.powerinfer.gguf", "--dry-run"])
    bench_out = capsys.readouterr().out

    assert smoke_rc == 0
    assert bench_rc == 0
    assert smoke_out == bench_out
