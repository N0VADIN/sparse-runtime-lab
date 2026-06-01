# Sparse Runtime Lab

Sparse Runtime Lab is a PowerInfer-first model tester and compatibility reporter. It turns the "Unsloth mindset" into a small, deterministic MVP: automate everything that can be automated, and put hard traffic-light gates around sparse-runtime risks.

## Why this exists

PowerInfer artifacts are not just normal GGUF files with a different name. A useful workflow needs to distinguish:

- dense GGUF baselines that should run in llama.cpp-compatible runtimes;
- `.powerinfer.gguf` artifacts that require PowerInfer sparse operators;
- sparse-friendly ReLU/ReGLU models that are likely good candidates;
- dense SwiGLU models that need recovery/evaluation gates before conversion is trusted.

This repository starts with **MVP 1: PowerInfer Model Tester**. It does not attempt blind SwiGLU-to-ReLU surgery. Instead it validates the runtime chain first.

## MVP capabilities

- Deterministic model intake analysis from artifact naming and extension.
- Compatibility scoring with traffic-light output:
  - 🟢 `PowerInfer-ready`
  - 🟡 `Experimental`
  - 🔴 `Not suitable`
- Runtime smoke-test command construction for PowerInfer or llama.cpp-style binaries.
- Coarse parsing for load success, first output/token, tokens/s, return code, timeout, and memory hints.
- Markdown reports that can be saved as CI artifacts.

## Install for local development

```bash
python -m pip install -e .
```

## Analyze a model artifact

```bash
sparse-runtime-lab analyze \
  --model Tiiny-SmallThinker-4BA0.6B-Instruct-Q4_K_M.powerinfer.gguf \
  --output report.md
```

## Run a PowerInfer smoke test

```bash
sparse-runtime-lab test \
  --runtime ./build/bin/main \
  --model model.powerinfer.gguf \
  --prompt "Explain sparse inference in one paragraph." \
  --tokens 128 \
  --threads 8 \
  --vram-budget 8 \
  --output report.md
```

The command returns `0` only when both the static model gate and runtime smoke test pass. It returns `2` for red static gates or failed runtime checks, which makes it suitable for CI.

## Intended roadmap

```text
Sparse Runtime Lab
│
├── Model Analyzer
│   ├── architecture detection
│   ├── activation detection
│   ├── tokenizer/chat-template check
│   └── compatibility score
│
├── Dense Exporter
│   ├── merge LoRA
│   ├── fp16 save
│   └── normal GGUF export
│
├── Sparse Profiler
│   ├── calibration dataset
│   ├── activation hooks
│   ├── sparsity stats
│   └── hot/cold neuron ranking
│
├── Sparse Converter
│   ├── ReLU/ReGLU recipe
│   ├── optional recovery fine-tune
│   ├── predictor/index generation
│   └── .powerinfer.gguf export
│
├── Eval Harness
│   ├── perplexity
│   ├── KL divergence
│   ├── top-k agreement
│   ├── prompt regression tests
│   └── chat-template sanity
│
└── Runtime Runner
    ├── llama.cpp dense baseline
    ├── PowerInfer sparse test
    ├── tok/s benchmark
    └── RAM/VRAM report
```

The guiding principle is **traffic lights, not magic**: a model should be called PowerInfer-ready only after it loads, produces output, keeps quality gates green, and beats the dense baseline on the target hardware.
