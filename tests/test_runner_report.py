import json
from pathlib import Path

import pytest

from sparse_runtime_lab.analyzer import analyze_model
from sparse_runtime_lab.models import Gate
from sparse_runtime_lab.report import final_gate, render_markdown_from_report, render_markdown_report
from sparse_runtime_lab.runner import build_command, parse_runtime_output, run_smoke_test
from sparse_runtime_lab.schema import artifact_report, dumps_report, load_report

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "runtime_logs"


def test_build_command_includes_vram_budget_and_extra_args():
    command = build_command(
        "./build/bin/main",
        "model.powerinfer.gguf",
        "hello",
        16,
        4,
        vram_budget=8,
        extra_args=("--temp", "0"),
    )

    assert command == (
        "./build/bin/main",
        "-m",
        "model.powerinfer.gguf",
        "-p",
        "hello",
        "-n",
        "16",
        "-t",
        "4",
        "--vram-budget",
        "8",
        "--temp",
        "0",
    )


@pytest.mark.parametrize("name", ["powerinfer_success", "powerinfer_failure", "llamacpp_success", "llamacpp_failure"])
def test_runtime_parsing_matches_log_fixtures(name):
    expected = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))[name]
    stdout = (FIXTURE_DIR / f"{name}.stdout").read_text(encoding="utf-8")
    stderr = (FIXTURE_DIR / f"{name}.stderr").read_text(encoding="utf-8")

    runtime = parse_runtime_output((name,), stdout=stdout, stderr=stderr, return_code=expected["return_code"])

    assert runtime.loaded is expected["loaded"]
    assert runtime.first_token is expected["first_token"]
    assert runtime.passed is expected["passed"]
    assert runtime.tokens_per_second == expected["tokens_per_second"]
    assert runtime.peak_memory_mb == expected["peak_memory_mb"]


def test_report_requires_runtime_success_for_green():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")
    runtime = parse_runtime_output(
        ("main",),
        stdout=(FIXTURE_DIR / "powerinfer_success.stdout").read_text(encoding="utf-8"),
        stderr=(FIXTURE_DIR / "powerinfer_success.stderr").read_text(encoding="utf-8"),
        return_code=0,
    )

    assert analysis.compatibility is Gate.YELLOW
    assert final_gate(analysis, None) is Gate.YELLOW
    assert final_gate(analysis, runtime) is Gate.GREEN
    report = render_markdown_report(analysis, runtime)
    assert "🟢 PowerInfer-ready" in report


def test_report_turns_runtime_failure_red():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")
    runtime = parse_runtime_output(
        ("main",),
        stdout=(FIXTURE_DIR / "powerinfer_failure.stdout").read_text(encoding="utf-8"),
        stderr=(FIXTURE_DIR / "powerinfer_failure.stderr").read_text(encoding="utf-8"),
        return_code=1,
    )

    assert analysis.compatibility is Gate.YELLOW
    assert final_gate(analysis, runtime) is Gate.RED
    report = render_markdown_report(analysis, runtime)
    assert "🔴 Not suitable" in report
    assert "Return code: `1`" in report


def test_json_schema_is_machine_readable_and_markdown_is_separate(tmp_path):
    analysis = analyze_model("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
    report = artifact_report(analysis)
    path = tmp_path / "report.json"
    path.write_text(dumps_report(report), encoding="utf-8")

    loaded = load_report(path)
    markdown = render_markdown_from_report(loaded)

    assert loaded["schema_version"] == 1
    assert loaded["report_type"] == "artifact"
    assert loaded["result"]["value"] == "yellow"
    assert loaded["runtime"] is None
    assert loaded["static_analysis"]["activation"] == "SwiGLU"
    assert "# Sparse Runtime Lab Report" in markdown


def test_missing_runtime_binary_is_failed_result_not_exception():
    runtime = run_smoke_test(("/definitely/missing/sparse-runtime-lab",), timeout_seconds=1)

    assert runtime.return_code == 127
    assert runtime.passed is False
    assert "No such file" in runtime.stderr_tail or "not found" in runtime.stderr_tail


def test_build_command_preserves_spaces_without_shell_string():
    command = build_command(
        "./runtime binary",
        "models/model with spaces.powerinfer.gguf",
        "prompt with spaces",
        8,
        2,
    )

    assert isinstance(command, tuple)
    assert command[0] == "./runtime binary"
    assert command[2] == "models/model with spaces.powerinfer.gguf"
    assert command[4] == "prompt with spaces"


def test_run_smoke_test_does_not_use_shell(monkeypatch):
    calls = []

    class Completed:
        stdout = "llm_load_tensors: model loaded\nhello\n"
        stderr = "12.5 tok/s\n"
        returncode = 0

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr("sparse_runtime_lab.runner.subprocess.run", fake_run)

    runtime = run_smoke_test(("runtime", "-m", "model.gguf"), timeout_seconds=5)

    assert runtime.passed is True
    assert calls[0][0] == ("runtime", "-m", "model.gguf")
    assert "shell" not in calls[0][1]


def test_run_smoke_test_timeout_is_failed_result(monkeypatch):
    import subprocess

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"], output="partial", stderr="timeout")

    monkeypatch.setattr("sparse_runtime_lab.runner.subprocess.run", fake_run)

    runtime = run_smoke_test(("runtime",), timeout_seconds=1)

    assert runtime.return_code == 124
    assert runtime.timed_out is True
    assert runtime.passed is False


def test_dense_llama_runtime_success_does_not_become_powerinfer_ready():
    analysis = analyze_model("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
    runtime = parse_runtime_output(
        ("llama.cpp",),
        stdout=(FIXTURE_DIR / "llamacpp_success.stdout").read_text(encoding="utf-8"),
        stderr=(FIXTURE_DIR / "llamacpp_success.stderr").read_text(encoding="utf-8"),
        return_code=0,
    )

    assert runtime.passed is True
    assert analysis.is_powerinfer_artifact is False
    assert final_gate(analysis, runtime) is Gate.YELLOW


def test_powerinfer_artifact_runtime_success_can_become_ready():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")
    runtime = parse_runtime_output(
        ("powerinfer",),
        stdout=(FIXTURE_DIR / "powerinfer_success.stdout").read_text(encoding="utf-8"),
        stderr=(FIXTURE_DIR / "powerinfer_success.stderr").read_text(encoding="utf-8"),
        return_code=0,
    )

    assert runtime.passed is True
    assert analysis.is_powerinfer_artifact is True
    assert final_gate(analysis, runtime) is Gate.GREEN


def test_load_only_stdout_does_not_count_as_first_token_evidence():
    runtime = parse_runtime_output(
        ("runtime",),
        stdout="llm_load_tensors: model loaded\n",
        stderr="",
        return_code=0,
    )

    assert runtime.loaded is True
    assert runtime.first_token is False
    assert runtime.passed is False


def test_zero_token_smoke_output_does_not_pass_readiness():
    runtime = parse_runtime_output(
        ("runtime", "-n", "0"),
        stdout="llm_load_tensors: model loaded\nPrompt echoed but no generation\n",
        stderr="llama_print_timings: load time = 100.00 ms\n",
        return_code=0,
    )

    assert runtime.loaded is True
    assert runtime.first_token is False
    assert runtime.passed is False


def test_zero_token_smoke_with_timing_does_not_pass_readiness():
    runtime = parse_runtime_output(
        ("runtime", "-n", "0"),
        stdout="llm_load_tensors: model loaded\n",
        stderr="llama_print_timings: eval time = 1.00 ms / 0 runs (0.00 tokens per second)\n",
        return_code=0,
    )

    assert runtime.loaded is True
    assert runtime.first_token is False
    assert runtime.passed is False
