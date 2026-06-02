import json
import sys

import pytest

from sparse_runtime_lab.cli import main
from sparse_runtime_lab.profiling import (
    ActivationProfileReport,
    LayerSparsitySummary,
    create_profiling_plan,
    activation_profile_report,
    dumps_profile_report,
    profiling_plan_report,
)
from sparse_runtime_lab.report import render_markdown_from_report


def test_profile_plan_schema_serialization(tmp_path):
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("prompt one\n", encoding="utf-8")
    plan = create_profiling_plan("model.gguf", calibration, 8, ("mlp", "ffn"))
    payload = profiling_plan_report(plan)

    assert payload["schema_version"] == 1
    assert payload["report_type"] == "profile_plan"
    assert payload["profile_plan"]["model_path"] == "model.gguf"
    assert payload["profile_plan"]["calibration"]["exists"] is True
    assert payload["profile_plan"]["target_modules"] == ["mlp", "ffn"]
    assert json.loads(dumps_profile_report(payload))["report_type"] == "profile_plan"


def test_activation_profile_report_skeleton_serialization(tmp_path):
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("prompt one\n", encoding="utf-8")
    plan = create_profiling_plan("model.gguf", calibration, 4, ("mlp",))
    report = ActivationProfileReport(
        plan=plan,
        layers=(LayerSparsitySummary(layer_index=0, module_name="mlp", total_values=10, zero_values=7, sparsity=0.7),),
    )

    payload = activation_profile_report(report)

    assert payload["report_type"] == "activation_profile"
    assert payload["layers"][0]["sparsity"] == 0.7


def test_profile_plan_rejects_invalid_sample_count(tmp_path):
    with pytest.raises(ValueError, match="max_samples"):
        create_profiling_plan("model.gguf", tmp_path / "calibration.txt", 0, ("mlp",))


def test_profile_plan_missing_calibration_path_warning(tmp_path):
    plan = create_profiling_plan("model.gguf", tmp_path / "missing.txt", 8, ("mlp",))
    payload = profiling_plan_report(plan)

    assert payload["profile_plan"]["calibration"]["exists"] is False
    assert "does not exist" in payload["profile_plan"]["calibration"]["warnings"][0]


def test_profile_plan_cli_outputs_json(tmp_path, capsys):
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("prompt one\n", encoding="utf-8")

    rc = main([
        "profile-plan",
        "--model",
        "model.gguf",
        "--calibration",
        str(calibration),
        "--max-samples",
        "3",
        "--target-modules",
        "mlp",
        "ffn",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["report_type"] == "profile_plan"
    assert payload["profile_plan"]["max_samples"] == 3
    assert payload["profile_plan"]["target_modules"] == ["mlp", "ffn"]


def test_profile_alias_matches_profile_plan(tmp_path, capsys):
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("prompt one\n", encoding="utf-8")

    rc_profile = main([
        "profile",
        "--model",
        "model.gguf",
        "--calibration",
        str(calibration),
        "--max-samples",
        "3",
        "--target-modules",
        "mlp",
        "ffn",
    ])
    profile_out = capsys.readouterr().out

    rc_profile_plan = main([
        "profile-plan",
        "--model",
        "model.gguf",
        "--calibration",
        str(calibration),
        "--max-samples",
        "3",
        "--target-modules",
        "mlp",
        "ffn",
    ])
    profile_plan_out = capsys.readouterr().out

    assert rc_profile == 0
    assert rc_profile_plan == 0
    assert profile_out == profile_plan_out


def test_profile_plan_markdown_rendering(tmp_path):
    plan = create_profiling_plan("model.gguf", tmp_path / "missing.txt", 3, ("mlp",))
    markdown = render_markdown_from_report(profiling_plan_report(plan))

    assert "# Activation Profiling Plan" in markdown
    assert "dry-run plan only" in markdown
    assert "Calibration exists: `no`" in markdown


def test_profile_plan_does_not_require_torch_or_transformers():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
