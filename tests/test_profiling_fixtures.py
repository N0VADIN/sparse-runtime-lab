import json
import sys
from pathlib import Path

from sparse_runtime_lab.cli import main
from sparse_runtime_lab.profiling import dumps_profile_report, validate_profile_report
from sparse_runtime_lab.report import render_markdown_from_report

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "profiling"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_profile_plan_fixture_round_trips_through_schema_helpers():
    fixture = _load_fixture("profile_plan_minimal.json")

    assert validate_profile_report(fixture) == ()
    round_tripped = json.loads(dumps_profile_report(fixture))
    assert round_tripped == fixture


def test_activation_profile_report_fixture_validates_expected_fields():
    fixture = _load_fixture("activation_profile_report_layers.json")

    assert validate_profile_report(fixture) == ()
    assert fixture["profile_plan"]["model_path"] == "model.gguf"
    assert len(fixture["layers"]) == 2
    assert fixture["layers"][0]["sparsity"] == 0.75


def test_retained_mass_curve_fixture_preserves_ordered_points():
    fixture = _load_fixture("retained_mass_curve.json")
    points = fixture["points"]

    assert [point["neurons"] for point in points] == [1, 2, 4]
    assert [point["retained_mass"] for point in points] == [0.25, 0.5, 0.9]


def test_invalid_profile_fixture_reports_clear_error():
    fixture = _load_fixture("invalid_profile_report_missing_layers.json")

    assert "activation_profile.layers must be a list" in validate_profile_report(fixture)


def test_markdown_renderer_includes_profile_fixture_fields_and_warnings():
    fixture = _load_fixture("activation_profile_report_layers.json")
    markdown = render_markdown_from_report(fixture)

    assert "# Activation Profiling Plan" in markdown
    assert "Model path: `model.gguf`" in markdown
    assert "Calibration path: `calibration.txt`" in markdown
    assert "Max samples: `3`" in markdown
    assert "Layer sparsity summaries" in markdown
    assert "Fixture-only profile" in markdown


def test_profile_plan_warning_fixture_renders_missing_calibration_warning():
    fixture = _load_fixture("profile_plan_with_warnings.json")
    markdown = render_markdown_from_report(fixture)

    assert "Calibration exists: `no`" in markdown
    assert "Calibration file does not exist" in markdown


def test_cli_renders_profile_fixture_to_markdown(tmp_path):
    output = tmp_path / "profile.md"

    rc = main(["report", "--input", str(FIXTURE_DIR / "activation_profile_report_layers.json"), "--output", str(output)])

    assert rc == 0
    assert "Layer `0`" in output.read_text(encoding="utf-8")


def test_fixture_paths_do_not_import_torch_or_transformers():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
