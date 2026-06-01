import pytest

from sparse_runtime_lab.analyzer import analyze_model
from sparse_runtime_lab.models import Gate


def test_powerinfer_relu_artifact_is_candidate_not_ready():
    analysis = analyze_model("Tiiny-SmallThinker-4BA0.6B-Instruct-Q4_K_M.powerinfer.gguf")

    assert analysis.compatibility is Gate.YELLOW
    assert analysis.is_powerinfer_artifact is True
    assert analysis.activation == "ReLU"
    assert analysis.quantization == "Q4_K_M"
    assert any("runtime evidence" in reason for reason in analysis.reasons)


def test_dense_swiglu_llama_is_experimental_candidate():
    analysis = analyze_model("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")

    assert analysis.compatibility is Gate.YELLOW
    assert analysis.activation == "SwiGLU"
    assert any("blind sparse conversion" in reason for reason in analysis.reasons)


def test_non_gguf_lora_adapter_is_red():
    analysis = analyze_model("my-qwen-lora-adapter.safetensors")

    assert analysis.compatibility is Gate.RED
    assert analysis.has_lora is True
    assert analysis.format == "safetensors"


def test_static_analysis_never_marks_models_ready():
    models = [
        "SmallThinker-Q4_K_M.powerinfer.gguf",
        "ReluLLaMA-7B-Q4_K_M.powerinfer.gguf",
        "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "unknown-model.gguf",
    ]

    assert all(analyze_model(model).compatibility is not Gate.GREEN for model in models)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("model-Q4_K_M.gguf", "Q4_K_M"),
        ("model-Q8_0.gguf", "Q8_0"),
        ("model-f16.gguf", "F16"),
    ],
)
def test_quantization_variants(model, expected):
    assert analyze_model(model).quantization == expected


def test_smallthinker_plain_gguf_is_not_powerinfer_artifact():
    analysis = analyze_model("SmallThinker-4BA0.6B-Instruct-Q4_K_M.gguf")

    assert analysis.family == "SmallThinker"
    assert analysis.is_powerinfer_artifact is False
    assert analysis.format == "GGUF"


def test_lora_adapter_gguf_is_red_not_candidate():
    analysis = analyze_model("qwen-lora-adapter.gguf")

    assert analysis.compatibility is Gate.RED
    assert analysis.has_lora is True
    assert any("merge before" in reason for reason in analysis.reasons)


def test_qlora_gguf_is_red():
    analysis = analyze_model("llama-qlora.gguf")

    assert analysis.compatibility is Gate.RED
    assert analysis.has_lora is True


def test_normal_qwen_dense_gguf_remains_candidate():
    analysis = analyze_model("qwen-7b-q4_k_m.gguf")

    assert analysis.compatibility is Gate.YELLOW
    assert analysis.has_lora is False


def test_powerinfer_artifact_without_lora_remains_candidate():
    analysis = analyze_model("SmallThinker-Q4_K_M.powerinfer.gguf")

    assert analysis.compatibility is Gate.YELLOW
    assert analysis.is_powerinfer_artifact is True
    assert analysis.has_lora is False


def test_powerinfer_artifact_with_lora_hint_is_red():
    analysis = analyze_model("SmallThinker-lora-Q4_K_M.powerinfer.gguf")

    assert analysis.compatibility is Gate.RED
    assert analysis.is_powerinfer_artifact is True
    assert analysis.has_lora is True
