"""Markdown reporting for Sparse Runtime Lab."""

from __future__ import annotations

from .models import Gate, ModelAnalysis, RuntimeResult


def final_gate(analysis: ModelAnalysis, runtime: RuntimeResult | None) -> Gate:
    """Combine static compatibility and optional runtime smoke-test status."""

    if runtime is not None and not runtime.passed:
        return Gate.RED
    return analysis.compatibility


def render_markdown_report(analysis: ModelAnalysis, runtime: RuntimeResult | None = None) -> str:
    """Render a concise traffic-light report."""

    gate = final_gate(analysis, runtime)
    lines = [
        "# Sparse Runtime Lab Report",
        "",
        f"**Result:** {gate.emoji} {gate.label}",
        "",
        "## Model intake",
        "",
        f"- Path: `{analysis.model_path}`",
        f"- Format: `{analysis.format}`",
        f"- Family: `{analysis.family}`",
        f"- Activation: `{analysis.activation}`",
        f"- Tokenizer: `{analysis.tokenizer}`",
        f"- Quantization: `{analysis.quantization}`",
        f"- LoRA/adapter hint: `{'yes' if analysis.has_lora else 'no'}`",
        f"- PowerInfer artifact hint: `{'yes' if analysis.is_powerinfer_artifact else 'no'}`",
        "",
        "## Gate reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in analysis.reasons or ("No deterministic warnings detected.",))

    if runtime is not None:
        lines.extend(
            [
                "",
                "## Runtime smoke test",
                "",
                f"- Command: `{' '.join(runtime.command)}`",
                f"- Loaded: `{'yes' if runtime.loaded else 'no'}`",
                f"- First token/output: `{'yes' if runtime.first_token else 'no'}`",
                f"- Tokens/s: `{_fmt(runtime.tokens_per_second)}`",
                f"- Peak memory MB: `{_fmt(runtime.peak_memory_mb)}`",
                f"- Return code: `{runtime.return_code}`",
                f"- Timed out: `{'yes' if runtime.timed_out else 'no'}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Runtime smoke test",
                "",
                "- Not run. Use `sparse-runtime-lab test --runtime ./build/bin/main --model model.powerinfer.gguf`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Next gates",
            "",
            "- Compare against a dense GGUF baseline before trusting quality deltas.",
            "- Add perplexity/KL/top-k agreement checks before approving converted SwiGLU models.",
            "- Verify chat template and EOS/BOS handling in both dense and sparse runtimes.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
