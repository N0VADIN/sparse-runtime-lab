"""Deterministic model intake analysis for GGUF and PowerInfer artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from .models import Gate, ModelAnalysis

POWERINFER_HINTS = ("powerinfer", ".powerinfer.gguf", "smallthinker", "relullama", "bamboo")
RELU_HINTS = ("relu", "reglu", "smallthinker", "bamboo")
SWIGLU_HINTS = ("llama", "qwen", "mistral", "gemma", "phi")
QUANT_PATTERN = re.compile(r"(?:^|[-_.])(q[2-8](?:_[a-z0-9]+)*(?:_[a-z0-9]+)?|f16|fp16|bf16|f32)(?:[-_.]|$)", re.I)


def analyze_model(path: str | Path) -> ModelAnalysis:
    """Analyze a model path without loading weights.

    This intentionally stays deterministic and fast: it infers compatibility from
    artifact naming, extension, and common model-family conventions. A future
    analyzer can augment this with GGUF metadata inspection while preserving the
    same traffic-light contract.
    """

    model_path = Path(path)
    name = model_path.name.lower()
    suffixes = "".join(model_path.suffixes).lower()

    is_gguf = name.endswith(".gguf")
    is_powerinfer = any(hint in name for hint in POWERINFER_HINTS)
    has_lora = any(marker in name for marker in ("lora", "adapter", "qlora"))

    family = _detect_family(name)
    activation = _detect_activation(name)
    quantization = _detect_quantization(name)
    tokenizer = _detect_tokenizer(family)
    fmt = "PowerInfer GGUF" if is_powerinfer and is_gguf else "GGUF" if is_gguf else suffixes.lstrip(".") or "unknown"

    reasons: list[str] = []
    if not is_gguf:
        reasons.append("Artifact is not a GGUF file; runtime smoke tests may need a converted export.")
    if is_powerinfer:
        reasons.append("PowerInfer-compatible naming hint detected.")
    if activation in {"ReLU", "ReGLU"}:
        reasons.append("Activation appears sparse-friendly.")
    if activation == "SwiGLU":
        reasons.append("Likely dense SwiGLU model; conversion requires recovery/eval gates.")
    if has_lora:
        reasons.append("LoRA/adapter hint detected; merge before dense or sparse runtime export.")

    compatibility = _score(is_gguf, is_powerinfer, activation, family)

    return ModelAnalysis(
        model_path=model_path,
        format=fmt,
        family=family,
        activation=activation,
        tokenizer=tokenizer,
        quantization=quantization,
        has_lora=has_lora,
        is_powerinfer_artifact=is_powerinfer,
        compatibility=compatibility,
        reasons=tuple(reasons),
    )


def _detect_family(name: str) -> str:
    display_names = {
        "smallthinker": "SmallThinker",
        "relullama": "ReluLLaMA",
        "bamboo": "Bamboo",
        "llama": "Llama",
        "qwen": "Qwen",
        "mistral": "Mistral",
        "gemma": "Gemma",
        "phi": "Phi",
    }
    for family, display_name in display_names.items():
        if family in name:
            return display_name
    return "unknown"


def _detect_activation(name: str) -> str:
    if "reglu" in name:
        return "ReGLU"
    if any(hint in name for hint in RELU_HINTS):
        return "ReLU"
    if any(hint in name for hint in SWIGLU_HINTS):
        return "SwiGLU"
    if "gelu" in name:
        return "GELU"
    return "unknown"


def _detect_quantization(name: str) -> str:
    match = QUANT_PATTERN.search(name)
    return match.group(1).upper() if match else "unknown"


def _detect_tokenizer(family: str) -> str:
    if family in {"Llama", "ReluLLaMA", "SmallThinker"}:
        return "llama-compatible"
    if family == "Qwen":
        return "qwen-compatible"
    if family == "Mistral":
        return "mistral-compatible"
    return "unknown"


def _score(is_gguf: bool, is_powerinfer: bool, activation: str, family: str) -> Gate:
    if not is_gguf:
        return Gate.RED
    if is_powerinfer and activation in {"ReLU", "ReGLU"}:
        return Gate.GREEN
    if family in {"SmallThinker", "ReluLLaMA", "Bamboo"}:
        return Gate.GREEN
    if activation in {"ReLU", "ReGLU"}:
        return Gate.YELLOW
    if activation == "SwiGLU":
        return Gate.YELLOW
    return Gate.RED
