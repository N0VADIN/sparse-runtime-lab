from sparse_runtime_lab.analyzer import analyze_model
from sparse_runtime_lab.models import Gate


def test_powerinfer_relu_artifact_is_green():
    analysis = analyze_model("Tiiny-SmallThinker-4BA0.6B-Instruct-Q4_K_M.powerinfer.gguf")

    assert analysis.compatibility is Gate.GREEN
    assert analysis.is_powerinfer_artifact is True
    assert analysis.activation == "ReLU"
    assert analysis.quantization == "Q4_K_M"


def test_dense_swiglu_llama_is_experimental():
    analysis = analyze_model("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")

    assert analysis.compatibility is Gate.YELLOW
    assert analysis.activation == "SwiGLU"
    assert any("conversion requires" in reason for reason in analysis.reasons)


def test_non_gguf_lora_adapter_is_red():
    analysis = analyze_model("my-qwen-lora-adapter.safetensors")

    assert analysis.compatibility is Gate.RED
    assert analysis.has_lora is True
    assert analysis.format == "safetensors"
