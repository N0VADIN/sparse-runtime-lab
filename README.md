# Sparse Runtime Lab

Sparse Runtime Lab is a small, deterministic lab for testing sparse LLM runtime artifacts, especially PowerInfer-style `.powerinfer.gguf` models.

It is intentionally glue code:

- local artifact inspection
- PowerInfer layout checks
- runtime smoke tests
- dense-vs-sparse report scaffolding
- reproducible JSON/Markdown reports
- future activation-sparsity profiling contracts

It does **not** reimplement PowerInfer, llama.cpp, Unsloth, TEAL, SparsingLaw, ProSparse, ReluLLaMA, sparse kernels, model surgery, or training workflows.

## Current status

The repository currently focuses on a conservative MVP 1 / MVP 1.5 foundation:

| Area | Status | Notes |
| --- | --- | --- |
| Static artifact analysis | Implemented | Local filename/path heuristics for GGUF, PowerInfer-style artifacts, family hints, activation hints, quantization, and LoRA/adapter hints. |
| PowerInfer layout checks | Implemented | Finds local executable candidates without building or running PowerInfer. |
| Runtime command construction | Implemented | Builds argument lists, not shell strings. |
| Runtime smoke parsing | Implemented | Parses coarse load/token/tok/s/memory/timeout evidence from runtime output fixtures or subprocess runs. |
| JSON/Markdown reports | Implemented | Reports are intended for CI artifacts and later dense-vs-sparse comparisons. |
| Activation profiling schemas | Planned | MVP 2A/2B/2C track; should remain optional and offline-testable first. |
| Real Torch profiling | Planned, optional | Must live behind an explicit `profiling` extra dependency. |
| Sparse conversion/export | Out of scope | No blind SwiGLU-to-ReLU surgery, no PowerInfer export, no training/fine-tuning. |

## Design constraints

- Keep the default package lightweight and deterministic.
- Do not add heavy ML dependencies by default.
- Do not auto-download models or datasets.
- Do not execute external binaries through shell strings.
- Keep static artifact analysis separate from runtime execution.
- Treat `PowerInfer-ready` as a gate that requires evidence, not a filename guess.
- Keep dense baseline success separate from sparse readiness.
- Prefer JSON reports first, with Markdown rendering for humans.
- Keep default tests CPU-only and offline: no GPU, PowerInfer install, llama.cpp install, model download, Torch, or Transformers required.

## Traffic-light semantics

Static analysis can identify candidates and hard blockers, but it cannot prove runtime readiness.

| Gate | Meaning |
| --- | --- |
| 🟢 `PowerInfer-ready` | Static gate is not red **and** runtime smoke-test evidence passes for the intended sparse artifact/runtime path. |
| 🟡 `Needs runtime evidence` | Static candidate or runnable layout, but readiness has not been proven. |
| 🔴 `Not suitable` | Static blocker, missing runtime layout, failed runtime smoke test, adapter-only artifact, or malformed report/input. |

Important implications:

- A dense GGUF baseline that passes llama.cpp smoke testing is **not** automatically PowerInfer-ready.
- A sparse-family name such as `SmallThinker`, `ReluLLaMA`, or `Bamboo` is not enough to prove a PowerInfer artifact.
- A file named `SmallThinker-Q4_K_M.powerinfer.gguf` is still only a candidate until it actually loads and produces generation evidence in the intended runtime.
- LoRA/adapter-like artifacts must be merged/exported before being treated as direct runtime candidates.
- Non-empty startup logs are not sufficient proof of generated tokens.

## Install for local development

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

Future real activation profiling must remain optional, for example:

```bash
python -m pip install -e ".[profiling]"
```

The base install must continue to work without Torch or Transformers.

## Static artifact analysis

`analyze` performs static artifact analysis only and writes a JSON report. It does not run a model or inspect remote repositories.

```bash
sparse-runtime-lab analyze \
  --model Tiiny-SmallThinker-4BA0.6B-Instruct-Q4_K_M.powerinfer.gguf \
  --output artifact-report.json
```

Render an existing JSON report to Markdown with the separate `report` command:

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

The CLI executes external runtimes with `subprocess.run([...], shell=False)`. It returns `0` only when the static model gate is not red and the runtime smoke test passes. It returns `2` for red static gates, missing layout/runtime failures, validation errors, or failed smoke tests.

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

## MVP 2 roadmap

MVP 2 is about activation-profiling readiness. It should be built in small, reviewable steps.

| Step | Scope | Rule |
| --- | --- | --- |
| MVP 2A | Profiling schema and plan layer | Define contracts first; no model loading. |
| MVP 2B | Offline/fixture-based profiling tests | Validate JSON/report flow with fixtures; no Torch/Transformers. |
| MVP 2C | Optional Torch profiler | Real profiling only behind an explicit extra dependency. |
| MVP 2 hardening | Contracts, fixtures, optional deps, CLI exit codes | Stabilize before adding more profiler features. |

Open planning issues:

- #10 — MVP 2B: offline fixture-based profiling tests
- #12 — MVP 2C: optional Torch activation profiler behind extra dependency
- #14 — MVP 2 hardening: profiling contracts, fixtures, optional deps, and regression gates

### MVP 2B: offline fixture-based tests

MVP 2B should add small, human-readable fixtures under something like:

```text
tests/fixtures/profiling/
  profile_plan_minimal.json
  profile_plan_with_warnings.json
  activation_profile_report_minimal.json
  activation_profile_report_layers.json
  retained_mass_curve.json
  invalid_profile_report_missing_layers.json
```

These tests must validate schema round-trips, warnings, invalid data, Markdown rendering, and CLI fixture rendering without importing Torch or Transformers.

### MVP 2C: optional Torch profiler

The first real profiler should remain optional and isolated. Importing `sparse_runtime_lab`, running MVP 1 commands, and rendering fixture reports must not require Torch/Transformers.

Expected behavior:

- `profile-run` fails clearly if profiling extras are missing.
- Local model paths are loaded only when explicitly requested.
- Calibration prompts come from local files.
- Hooks collect summaries, not huge tensors by default.
- Output uses the existing activation profile JSON contract.
- No sparse conversion, export, training, or downloads are introduced.

## Agent workflow notes

When using Codex or another coding agent, prefer small sequential tasks.

Recommended modes:

```text
Fix mode:
  Explain root cause.
  Apply the smallest fix.
  Do not add features.
  Run focused tests, then the full suite.

Test scope mode:
  Define the public contract.
  List happy paths, edge cases, failure modes, and regressions.
  Return the test matrix before coding.

Hardening mode:
  Stabilize contracts, fixtures, reports, and dependency boundaries.
  Do not add unrelated profiler features.
```

Avoid parallel feature branches that cover the same scope. If a later PR supersedes an earlier MVP PR, close the older PR instead of rebasing it unless it contains unique work.

## Intended architecture

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
└── Profiling/Eval Harness
    ├── profiling plan JSON
    ├── offline profiling fixtures
    ├── optional Torch activation profiler
    ├── retained-mass curves
    ├── perplexity/KL/top-k agreement
    └── prompt regression tests
```

## License

This project is licensed under the Apache License 2.0. External runtimes, model weights, datasets, and generated artifacts keep their own licenses and terms.
