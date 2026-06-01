import json

from sparse_runtime_lab.analyzer import analyze_model
from sparse_runtime_lab.models import Gate
from sparse_runtime_lab.report import final_gate, render_json_report, render_markdown_report
from sparse_runtime_lab.runner import build_command, parse_runtime_output, run_smoke_test


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


def test_parse_runtime_output_extracts_metrics():
    runtime = parse_runtime_output(
        ("main",),
        stdout="llm_load_tensors: model loaded\nHello sparse world\n",
        stderr="eval time = 42.00 ms / 128 runs (55.5 tokens per second)\npeak memory 1.5 GB\n",
        return_code=0,
    )

    assert runtime.passed is True
    assert runtime.tokens_per_second == 55.5
    assert runtime.peak_memory_mb == 1536


def test_report_requires_runtime_success_for_green():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")
    runtime = parse_runtime_output(
        ("main",),
        stdout="llm_load_tensors: model loaded\nhello\n",
        stderr="54.2 tok/s\n",
        return_code=0,
    )

    assert analysis.compatibility is Gate.YELLOW
    assert final_gate(analysis, None) is Gate.YELLOW
    assert final_gate(analysis, runtime) is Gate.GREEN
    report = render_markdown_report(analysis, runtime)
    assert "🟢 PowerInfer-ready" in report


def test_report_turns_runtime_failure_red():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")
    runtime = parse_runtime_output(("main",), stdout="", stderr="crash", return_code=1)

    assert analysis.compatibility is Gate.YELLOW
    assert final_gate(analysis, runtime) is Gate.RED
    report = render_markdown_report(analysis, runtime)
    assert "🔴 Not suitable" in report
    assert "Return code: `1`" in report


def test_json_report_is_machine_readable():
    analysis = analyze_model("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
    payload = json.loads(render_json_report(analysis))

    assert payload["schema_version"] == 1
    assert payload["result"]["value"] == "yellow"
    assert payload["runtime"] is None
    assert payload["static_analysis"]["activation"] == "SwiGLU"


def test_missing_runtime_binary_is_failed_result_not_exception():
    runtime = run_smoke_test(("/definitely/missing/sparse-runtime-lab",), timeout_seconds=1)

    assert runtime.return_code == 127
    assert runtime.passed is False
    assert "No such file" in runtime.stderr_tail or "not found" in runtime.stderr_tail
