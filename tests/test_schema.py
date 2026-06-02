import json

import pytest

from sparse_runtime_lab.analyzer import analyze_model
from sparse_runtime_lab.layout import check_powerinfer_layout
from sparse_runtime_lab.schema import artifact_report, dumps_report, layout_report, load_report


def test_artifact_schema_includes_runtime_fields():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")
    report = artifact_report(analysis, planned_command=("runtime", "-m", "model.powerinfer.gguf"))

    assert report["schema_version"] == 1
    assert report["report_type"] == "artifact"
    assert report["planned_command"] == ["runtime", "-m", "model.powerinfer.gguf"]
    assert report["runtime"] is None


def test_layout_schema_report_has_layout_fields(tmp_path):
    check = check_powerinfer_layout(tmp_path / "missing")
    report = layout_report(check)

    assert report["report_type"] == "layout"
    assert report["layout"]["root"] == str(tmp_path / "missing")
    assert report["layout"]["executable"] is None


def test_schema_dump_is_stable_json_with_trailing_newline():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")
    dumped = dumps_report(artifact_report(analysis))

    assert dumped.endswith("\n")
    assert json.loads(dumped)["schema_version"] == 1


def test_load_report_rejects_non_object(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="object"):
        load_report(path)


def test_load_report_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"schema_version": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_report(path)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"schema_version": 1, "report_type": "artifact", "result": {}}, "static_analysis"),
        ({"schema_version": 1, "report_type": "layout", "result": {}}, "layout"),
        ({"schema_version": 1, "report_type": "profile_plan"}, "profile_plan"),
        ({"schema_version": 1, "report_type": "activation_profile", "profile_plan": {}}, "layers"),
    ],
)
def test_load_report_rejects_missing_required_top_level_keys(tmp_path, payload, expected):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        load_report(path)
