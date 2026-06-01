"""PowerInfer artifact layout and command helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shlex


@dataclass(frozen=True)
class PowerInferLayout:
    """Summary of a directory containing PowerInfer-style artifacts."""

    root: str
    powerinfer_gguf: list[str]
    quantized_powerinfer_gguf: list[str]
    activation_files: list[str]
    gpu_index_files: list[str]
    warnings: list[str]

    @property
    def is_runnable_candidate(self) -> bool:
        return bool(self.powerinfer_gguf or self.quantized_powerinfer_gguf)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["is_runnable_candidate"] = self.is_runnable_candidate
        return data


def inspect_powerinfer_dir(path: str | Path) -> PowerInferLayout:
    """Inspect a directory for common PowerInfer GGUF artifacts."""

    root = Path(path)
    warnings: list[str] = []

    if not root.exists():
        raise FileNotFoundError(f"PowerInfer directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"PowerInfer path is not a directory: {root}")

    all_files = [p for p in root.rglob("*") if p.is_file()]

    powerinfer_gguf = sorted(
        _relative(p, root)
        for p in all_files
        if p.name.endswith(".powerinfer.gguf") and not p.name.endswith(".q4.powerinfer.gguf")
    )
    quantized_powerinfer_gguf = sorted(
        _relative(p, root) for p in all_files if p.name.endswith(".q4.powerinfer.gguf")
    )
    activation_files = sorted(
        _relative(p, root)
        for p in all_files
        if "activation" in {part.lower() for part in p.relative_to(root).parts}
    )
    gpu_index_files = sorted(
        _relative(p, root) for p in all_files if p.name.endswith(".generated.gpuidx")
    )

    if not powerinfer_gguf and not quantized_powerinfer_gguf:
        warnings.append("No .powerinfer.gguf or .q4.powerinfer.gguf files found.")
    if not activation_files:
        warnings.append("No activation/ profile files found; some workflows may need them.")
    if not gpu_index_files:
        warnings.append("No generated GPU index files found; runtime may generate them on first run.")

    return PowerInferLayout(
        root=str(root),
        powerinfer_gguf=powerinfer_gguf,
        quantized_powerinfer_gguf=quantized_powerinfer_gguf,
        activation_files=activation_files,
        gpu_index_files=gpu_index_files,
        warnings=warnings,
    )


def build_powerinfer_smoke_command(
    *,
    binary: str | Path,
    model: str | Path,
    prompt: str,
    tokens: int = 128,
    threads: int = 8,
    vram_budget: int | float | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build a PowerInfer smoke-test command as argv, not a shell string."""

    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if vram_budget is not None and vram_budget < 0:
        raise ValueError("vram_budget must be non-negative")

    command = [
        str(binary),
        "-m",
        str(model),
        "-n",
        str(tokens),
        "-t",
        str(threads),
        "-p",
        prompt,
    ]
    if vram_budget is not None:
        command.extend(["--vram-budget", str(vram_budget)])
    if extra_args:
        command.extend(extra_args)
    return command


def shell_quote_command(command: list[str]) -> str:
    """Render argv as a copy-pasteable shell command for humans."""

    return " ".join(shlex.quote(part) for part in command)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
