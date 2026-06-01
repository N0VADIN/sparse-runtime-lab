# Sparse Runtime Lab

Sparse Runtime Lab is a small, deterministic lab for testing sparse LLM runtime artifacts, especially PowerInfer-style `.powerinfer.gguf` models. It is intentionally glue code: artifact inspection, PowerInfer layout checks, runtime smoke tests, dense-vs-sparse report scaffolding, and reproducible JSON/Markdown reports.

It does **not** reimplement PowerInfer, llama.cpp, Unsloth, TEAL, SparsingLaw, ProSparse, ReluLLaMA, sparse kernels, or model surgery.

## Design constraints

- Keep MVP 0/1 small and deterministic.
- Do not add heavy ML dependencies by default.
- Do not auto-download models.
- Do not execute external binaries through shell strings.
- Keep static artifact analysis separate from runtime execution.
- Treat `PowerInfer-ready` as a gate that requires runtime evidence, not a filename guess.
- Prefer JSON + Markdown reports.
- Keep tests CPU-only: no GPU, PowerInfer install, or model downloads required.

## MVP capabilities

- Static artifact analysis from local path/name conventions:
  - GGUF vs PowerInfer-style GGUF hints;
  - family hints such as SmallThinker, ReluLLaMA, Bamboo, Llama, Qwen, Mistral;
  - activation hints such as ReLU/ReGLU/SwiGLU/GELU;
  - quantization and LoRA/adapter hints.
- PowerInfer directory/layout checks that find a local executable without running it.
- Safe runtime command construction as argument lists, not shell strings.
- Runtime smoke-test parsing for load success, first output/token evidence, tokens/s, memory hints, return code, and timeout.
- Markdown and JSON reports suitable for CI artifacts.

## Traffic-light semantics

Static analysis can identify candidates and hard blockers, but it cannot prove readiness.

| Gate | Meaning |
| --- | --- |
| 🟢 `PowerInfer-ready` | Static gate is not red **and** runtime smoke-test evidence passes. |
| 🟡 `Needs runtime evidence` | Static candidate or runnable layout, but readiness has not been proven. |
| 🔴 `Not suitable` | Static blocker, missing runtime layout, or failed runtime smoke test. |

This means a file named `SmallThinker-Q4_K_M.powerinfer.gguf` is still only yellow until it actually loads and produces output in the intended runtime.

## Install for local development

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

## Static artifact analysis

`analyze` performs static artifact analysis only and writes a JSON report. It does not run a model or inspect remote repositories.

```bash
sparse-runtime-lab analyze \
  --model Tiiny-SmallThinker-4BA0.6B-Instruct-Q4_K_M.powerinfer.gguf \
  --output artifact-report.json
```

Render the existing JSON report to Markdown with the separate `report` command:

```bash
sparse-runtime-lab report \
  --input artifact-report.json \
  --output artifact-report.md
```

## PowerInfer layout check

Use this before smoke testing to verify that a local checkout/build has a runnable binary. This command does not build or execute PowerInfer.

```bash
sparse-runtime-lab check-layout \
  --powerinfer-dir ./PowerInfer \
  --output powerinfer-layout.json
```

## Runtime smoke test

`smoke --dry-run` builds and records the exact command without executing the runtime:

```bash
sparse-runtime-lab smoke \
  --runtime ./PowerInfer/build/bin/main \
  --model model.powerinfer.gguf \
  --tokens 128 \
  --threads 8 \
  --vram-budget 8 \
  --dry-run \
  --output smoke-plan.json
```

Run the smoke test by omitting `--dry-run`:

```bash
sparse-runtime-lab smoke \
  --runtime ./PowerInfer/build/bin/main \
  --model model.powerinfer.gguf \
  --prompt "Explain sparse inference in one paragraph." \
  --tokens 128 \
  --threads 8 \
  --vram-budget 8 \
  --output smoke-report.json
```

The CLI executes external runtimes with `subprocess.run([...], shell=False)`. It returns `0` only when the static model gate is not red and the runtime smoke test passes. It returns `2` for red static gates, missing layout/runtime failures, or failed smoke tests.

## Dense vs sparse workflow intent

MVP 1.5 validates the runtime chain first. A safe sparse workflow should still compare against a dense baseline before making quality or speed claims:

```text
Dense GGUF baseline
  → llama.cpp smoke test
  → prompt/output sanity report

PowerInfer sparse artifact
  → PowerInfer layout check
  → PowerInfer smoke test
  → compare load/output/tok/s/memory against dense baseline
```

Blind SwiGLU-to-ReLU conversion is intentionally out of scope. Future sparse conversion experiments need recovery training and quality-aware gates such as perplexity, KL divergence, top-k agreement, prompt regression tests, and chat-template/EOS sanity checks.

## Intended roadmap

```text
Sparse Runtime Lab
│
├── Model Analyzer
│   ├── local artifact detection
│   ├── architecture/activation metadata readers
│   ├── tokenizer/chat-template checks
│   └── compatibility score
│
├── Runtime Layout Checker
│   ├── PowerInfer checkout/build validation
│   └── llama.cpp dense baseline validation
│
├── Runtime Runner
│   ├── llama.cpp dense baseline
│   ├── PowerInfer sparse smoke test
│   ├── tok/s parser
│   └── RAM/VRAM parser
│
├── Comparison Reporter
│   ├── JSON reports
│   ├── Markdown summaries
│   └── dense-vs-sparse deltas
│
└── Future Profiling/Eval Harness
    ├── training-free activation sparsity profiling
    ├── retained-mass curves
    ├── perplexity/KL/top-k agreement
    └── prompt regression tests
```
