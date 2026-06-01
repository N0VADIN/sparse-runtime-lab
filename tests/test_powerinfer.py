from __future__ import annotations

from sparse_runtime_lab.powerinfer import (
    build_powerinfer_smoke_command,
    inspect_powerinfer_dir,
    shell_quote_command,
)


def test_inspect_powerinfer_dir_detects_expected_files(tmp_path):
    (tmp_path / "activation").mkdir()
    (tmp_path / "model.powerinfer.gguf").write_text("fake")
    (tmp_path / "model.q4.powerinfer.gguf").write_text("fake")
    (tmp_path / "activation" / "activation_0.pt").write_text("fake")
    (tmp_path / "model.powerinfer.gguf.generated.gpuidx").write_text("fake")

    layout = inspect_powerinfer_dir(tmp_path)

    assert layout.is_runnable_candidate
    assert layout.powerinfer_gguf == ["model.powerinfer.gguf"]
    assert layout.quantized_powerinfer_gguf == ["model.q4.powerinfer.gguf"]
    assert layout.activation_files == ["activation/activation_0.pt"]
    assert layout.gpu_index_files == ["model.powerinfer.gguf.generated.gpuidx"]
    assert layout.warnings == []


def test_inspect_powerinfer_dir_warns_on_missing_optional_files(tmp_path):
    (tmp_path / "model.powerinfer.gguf").write_text("fake")

    layout = inspect_powerinfer_dir(tmp_path)

    assert layout.is_runnable_candidate
    assert any("activation" in warning for warning in layout.warnings)
    assert any("GPU index" in warning for warning in layout.warnings)


def test_build_powerinfer_smoke_command():
    command = build_powerinfer_smoke_command(
        binary="./main",
        model="model.powerinfer.gguf",
        prompt="hello sparse world",
        tokens=16,
        threads=4,
        vram_budget=8,
        extra_args=["--temp", "0.2"],
    )

    assert command == [
        "./main",
        "-m",
        "model.powerinfer.gguf",
        "-n",
        "16",
        "-t",
        "4",
        "-p",
        "hello sparse world",
        "--vram-budget",
        "8",
        "--temp",
        "0.2",
    ]
    assert "hello sparse world" in shell_quote_command(command)
