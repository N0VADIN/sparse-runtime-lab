"""Minimal GGUF inspection helpers.

This module intentionally parses only the magic/version/header count fields needed for
safe preflight checks. It does not try to be a full GGUF reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import struct

GGUF_MAGIC = b"GGUF"
_MIN_HEADER_BYTES = 24


@dataclass(frozen=True)
class GgufHeader:
    """Small summary of a GGUF file header."""

    path: str
    size_bytes: int
    version: int
    tensor_count: int
    metadata_kv_count: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class GgufInspectionError(ValueError):
    """Raised when a file is not inspectable as GGUF."""


def inspect_gguf_header(path: str | Path) -> GgufHeader:
    """Inspect the fixed GGUF header fields.

    GGUF v2/v3 store the version, tensor count, and metadata KV count directly
    after the magic bytes in little-endian order:

    - magic: 4 bytes
    - version: uint32
    - tensor_count: uint64
    - metadata_kv_count: uint64
    """

    file_path = Path(path)
    if not file_path.exists():
        raise GgufInspectionError(f"GGUF file does not exist: {file_path}")
    if not file_path.is_file():
        raise GgufInspectionError(f"GGUF path is not a file: {file_path}")

    size = file_path.stat().st_size
    if size < _MIN_HEADER_BYTES:
        raise GgufInspectionError(
            f"File is too small to contain a GGUF header: {file_path} ({size} bytes)"
        )

    with file_path.open("rb") as handle:
        header = handle.read(_MIN_HEADER_BYTES)

    magic, version, tensor_count, metadata_kv_count = struct.unpack("<4sIQQ", header)
    if magic != GGUF_MAGIC:
        raise GgufInspectionError(f"File does not start with GGUF magic: {file_path}")
    if version <= 0:
        raise GgufInspectionError(f"Unsupported GGUF version {version}: {file_path}")

    return GgufHeader(
        path=str(file_path),
        size_bytes=size,
        version=version,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
    )
