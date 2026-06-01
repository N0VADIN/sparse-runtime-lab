# MVP 2 Activation Profiler Plan

MVP 2A adds only the activation-profiling **schema and planning layer**. It does not load real models, import ML frameworks, execute runtimes, download calibration data, or perform sparse conversion.

## MVP 2A scope now in the package

- `sparse_runtime_lab.profiling.schema` defines JSON-compatible dataclasses for:
  - `ProfilingPlan`
  - `CalibrationSource`
  - `LayerSparsitySummary`
  - `ActivationProfileReport`
- `sparse_runtime_lab.profiling.plan` creates dry-run profiling plans from local paths and target module names.
- `sparse-runtime-lab profile-plan` emits a JSON dry-run plan.
- Markdown rendering can summarize `profile_plan` and `activation_profile` report skeletons.

## MVP 2A non-goals

- No `torch` dependency.
- No `transformers` dependency.
- No model loading.
- No calibration file reading beyond existence checks.
- No model downloads.
- No sparse kernels or sparse conversion.
- No runtime binary execution.

## Example dry-run plan

```bash
sparse-runtime-lab profile-plan \
  --model model.gguf \
  --calibration calibration_prompts.txt \
  --max-samples 128 \
  --target-modules mlp ffn \
  --output profile-plan.json
```

The command records intent and warnings, for example when the calibration path is missing, but it remains a planning step only.

## Future MVP 2B goals

- Measure activation sparsity on local calibration prompts.
- Produce per-layer summary JSON files and retained-mass curves.
- Keep quality-aware gates separate from runtime smoke tests.
- Make profiling reproducible enough to compare dense and sparse candidates.

## Future optional dependency boundary

A later extra can hold heavyweight profiling dependencies:

```toml
[project.optional-dependencies]
profiler = [
  "torch>=2",
  "transformers>=4",
]
```

The default `sparse-runtime-lab` install should continue to run analyzer, layout, smoke, profile planning, report rendering, and tests without these packages.

## Future workflow

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

## Future output artifacts

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

## Suggested future gates

- Calibration prompt count and token count are above configured minimums.
- No NaNs or infs in activation summaries.
- Per-layer sparsity is reported with confidence intervals or raw sample counts.
- Retained-mass thresholds are explicit and reproducible.
- Any conversion recommendation remains yellow until quality metrics pass.

## Open questions

- Which local metadata reader should become the canonical source for chat template and EOS/BOS checks?
- Should measured profiler outputs share the same schema family as smoke reports, or remain in a separate `activation_profile` report type?
- Which small public calibration sets are acceptable if downloads are explicitly enabled in a future version?
