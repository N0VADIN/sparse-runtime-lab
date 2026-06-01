import pytest

from sparse_runtime_lab.analyzer import analyze_model
from sparse_runtime_lab.report import render_markdown_from_report
from sparse_runtime_lab.schema import artifact_report


def test_markdown_renders_planned_smoke_command():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")
    report = artifact_report(analysis, planned_command=("runtime", "-m", "model.powerinfer.gguf"))

    markdown = render_markdown_from_report(report)

    assert "## Planned smoke command" in markdown
    assert "runtime -m model.powerinfer.gguf" in markdown


def test_markdown_rejects_unknown_report_type():
    with pytest.raises(ValueError, match="unsupported report_type"):
        render_markdown_from_report({"schema_version": 1, "report_type": "unknown"})
