# Third-party notes

Sparse Runtime Lab is an independent Apache-2.0 project. It may interoperate with external tools, runtimes, models, and datasets. Those external artifacts retain their own licenses and terms.

## Runtime and tooling projects

| Project | Typical role | Notes |
| --- | --- | --- |
| PowerInfer | Sparse inference runtime and PowerInfer GGUF tooling | Do not copy runtime code into this repo without preserving upstream notices and checking license compatibility. Prefer invoking it as an external binary or adapter. |
| llama.cpp / ggml | Dense GGUF reference runtime and GGUF ecosystem | Use as a dense baseline and GGUF reference. Preserve upstream notices if code is reused. |
| Unsloth | Fine-tuning and GGUF export workflows | This project is inspired by the guided workflow style. It is not affiliated with Unsloth. Be especially careful with code from differently licensed subdirectories. |
| TEAL, SparsingLaw, R-Sparse, dynamic-sparsity, Polar-Sparsity | Research references for activation sparsity, profiling, metrics, and kernels | Treat as references unless their code is explicitly imported with license review. |

## Model weights and datasets

Model weights, tokenizer files, calibration prompts, benchmark datasets, and generated artifacts are not automatically covered by this repository's Apache-2.0 license.

Before publishing reports or generated artifacts, record:

- source model repository
- model license
- dataset/calibration source
- quantization format
- conversion commands
- runtime version
- hardware summary

## Dependency policy

Initial MVP code should prefer the Python standard library. Optional integrations can be added later behind extras, for example:

- `powerinfer`
- `llama-cpp`
- `torch`
- `transformers`
- `eval`

The default install should stay lightweight and not download models or execute external binaries implicitly.
