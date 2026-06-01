from __future__ import annotations

import struct

import pytest

from sparse_runtime_lab.gguf import GgufInspectionError, inspect_gguf_header


def test_inspect_gguf_header(tmp_path):
    path = tmp_path / "model.powerinfer.gguf"
    path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 42, 7) + b"payload")

    header = inspect_gguf_header(path)

    assert header.version == 3
    assert header.tensor_count == 42
    assert header.metadata_kv_count == 7
    assert header.size_bytes > 24


def test_rejects_non_gguf(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"nope" * 8)

    with pytest.raises(GgufInspectionError):
        inspect_gguf_header(path)
