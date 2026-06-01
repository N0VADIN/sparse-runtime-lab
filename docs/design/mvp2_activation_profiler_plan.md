# MVP 2 Activation Profiler Plan

MVP 1.5 stops at artifact inspection, layout checks, runtime smoke tests, and reproducible reports. MVP 2 can add activation-sparsity profiling, but only as an optional layer that keeps the default package lightweight.

## Goals

- Measure activation sparsity on local calibration prompts.
- Produce per-layer summary JSON files and retained-mass curves.
- Keep quality-aware gates separate from runtime smoke tests.
- Make profiling reproducible enough to compare dense and sparse candidates.

## Non-goals

- No sparse kernel implementation.
- No PowerInfer or llama.cpp reimplementation.
- No blind SwiGLU-to-ReLU conversion.
- No default `torch` or `transformers` dependency in the base package.
- No model downloads unless a future CLI flag explicitly requests them.

## Proposed optional dependency boundary

A future extra can hold heavyweight profiling dependencies:

```toml
[project.optional-dependencies]
profiler = [
  "torch>=2",
  "transformers>=4",
]
```

The default `sparse-runtime-lab` install should continue to run analyzer, layout, smoke, report, and tests without these packages.

## Proposed workflow

```text
calibration prompts
  → local model load through optional profiler extra
  → forward hooks on FFN activations
  → per-layer sparsity stats
  → hot/cold neuron ranking
  → retained-mass curves
  → JSON report
  → Markdown summary
```

## Output artifacts

```text
profile/
├── metadata.json
├── global_summary.json
├── layers/
│   ├── layer_000.json
│   ├── layer_001.json
│   └── ...
└── plots/
    ├── sparsity_by_layer.png
    └── retained_mass_vs_neurons.png
```

## Suggested gates

- Calibration prompt count and token count are above configured minimums.
- No NaNs or infs in activation summaries.
- Per-layer sparsity is reported with confidence intervals or raw sample counts.
- Retained-mass thresholds are explicit and reproducible.
- Any conversion recommendation remains yellow until quality metrics pass.

## Open questions

- Which local metadata reader should become the canonical source for chat template and EOS/BOS checks?
- Should profiler outputs be compared directly in the same schema family as smoke reports, or kept in a separate `profile` report type?
- Which small public calibration sets are acceptable if downloads are explicitly enabled in a future version?
