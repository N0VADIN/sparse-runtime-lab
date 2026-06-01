# Sparse Runtime Lab

Unsloth-inspired tooling for profiling, inspecting, and validating activation-sparse LLM runtime artifacts.

The initial scope is intentionally small and practical:

- inspect GGUF and PowerInfer-style model artifacts
- validate expected `.powerinfer.gguf` repository layouts
- build reproducible PowerInfer smoke-test commands
- capture dense-vs-sparse runtime reports later without hiding failures

This repository is **not** an official Unsloth, PowerInfer, llama.cpp, or Tiiny AI project. It is an independent lab for experimenting with sparse inference workflows.

## MVP roadmap

### MVP 0: PowerInfer model zoo smoke tester

Goal: verify that known `.powerinfer.gguf` artifacts can be discovered, inspected, and launched with a reproducible command.

Planned checks:

- detect `.powerinfer.gguf` and optional `.q4.powerinfer.gguf` files
- detect optional `activation/` profile artifacts
- detect generated GPU index files
- build PowerInfer command lines without executing shell strings
- write JSON reports for later comparison

### MVP 1: Sparse profiler skeleton

Goal: add a clean place for activation-sparsity profiling without pretending that arbitrary SwiGLU models are automatically PowerInfer-ready.

Planned checks:

- architecture and activation-function detection
- calibration-prompt handling
- per-layer sparsity summaries
- hot/cold neuron statistics
- quality gates before any sparse export is trusted

## Installation for development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## CLI examples

Inspect a GGUF header:

```bash
srl inspect-gguf /models/example.powerinfer.gguf --json
```

Inspect a PowerInfer-style model directory:

```bash
srl inspect-powerinfer-dir /models/ReluLLaMA-7B-PowerInfer-GGUF --json
```

Print a PowerInfer command without running it:

```bash
srl run-smoke \
  --binary ./PowerInfer/build/bin/main \
  --model /models/example.powerinfer.gguf \
  --prompt "Explain activation sparsity in one paragraph." \
  --tokens 128 \
  --threads 8 \
  --vram-budget 8 \
  --dry-run
```

## Design principles

- Prefer boring, reproducible checks over magic conversions.
- Keep runtime integrations as adapters, not copied upstream code.
- Treat model weights and datasets as separately licensed artifacts.
- Fail loudly when an artifact is ambiguous or incomplete.
- Validate quality and speed separately.

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.

External tools and model weights keep their own licenses. See `THIRD_PARTY.md`.
