"""Read GGUF descriptors and resolve locally installed Ollama model blobs.

Only metadata and tensor descriptors are read. Tensor data—and therefore
actual weight values—never enters memory.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


class GGUFError(RuntimeError):
    """A user-facing error raised for invalid or unsupported model input."""


@dataclass(frozen=True)
class ArrayInfo:
    """A skipped GGUF array represented by element type and item count."""

    element_type: int
    count: int


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def type_name(self) -> str:
        return TENSOR_TYPES.get(self.type_id, f"type-{self.type_id}")


@dataclass(frozen=True)
class GGUFModel:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: tuple[TensorInfo, ...]

    def tensor(self, name: str) -> TensorInfo | None:
        return next((tensor for tensor in self.tensors if tensor.name == name), None)


VALUE_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
STRING = 8
ARRAY = 9

TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
}


class _Reader:
    """Small binary reader for the GGUF v2/v3 header format."""

    def __init__(self, source: BinaryIO):
        self.source = source

    def exact(self, size: int) -> bytes:
        data = self.source.read(size)
        if len(data) != size:
            raise GGUFError("unexpected end of GGUF header")
        return data

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.exact(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        size = self.unpack("<Q")
        if size > 256 * 1024 * 1024:
            raise GGUFError(f"refusing implausible GGUF string length {size}")
        try:
            return self.exact(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise GGUFError("GGUF header contains invalid UTF-8") from error

    def value(self, value_type: int) -> Any:
        if value_type in VALUE_FORMATS:
            return self.unpack(VALUE_FORMATS[value_type])
        if value_type == STRING:
            return self.string()
        if value_type != ARRAY:
            raise GGUFError(f"unsupported GGUF metadata type {value_type}")

        element_type = self.unpack("<I")
        count = self.unpack("<Q")
        if count > 1_000_000_000:
            raise GGUFError(f"refusing implausible GGUF array length {count}")
        if element_type == ARRAY:
            raise GGUFError("nested GGUF arrays are unsupported")
        self.skip_values(element_type, count)
        return ArrayInfo(element_type, count)

    def skip_values(self, value_type: int, count: int) -> None:
        if value_type in VALUE_FORMATS:
            size = struct.calcsize(VALUE_FORMATS[value_type]) * count
            self.source.seek(size, 1)
            return
        if value_type == STRING:
            for _ in range(count):
                size = self.unpack("<Q")
                if size > 256 * 1024 * 1024:
                    raise GGUFError(
                        f"refusing implausible GGUF string length {size}"
                    )
                self.source.seek(size, 1)
            return
        raise GGUFError(f"unsupported GGUF array element type {value_type}")


def read_gguf(path: str | Path) -> GGUFModel:
    """Read metadata and tensor shapes without reading tensor data."""
    model_path = Path(path)
    try:
        with model_path.open("rb") as source:
            reader = _Reader(source)
            if reader.exact(4) != b"GGUF":
                raise GGUFError(f"{model_path} is not a GGUF file")
            version = reader.unpack("<I")
            if version not in {2, 3}:
                raise GGUFError(
                    f"unsupported GGUF version {version}; expected 2 or 3"
                )
            tensor_count = reader.unpack("<Q")
            metadata_count = reader.unpack("<Q")
            if tensor_count > 10_000_000 or metadata_count > 10_000_000:
                raise GGUFError("GGUF header declares implausibly many entries")

            metadata: dict[str, Any] = {}
            for _ in range(metadata_count):
                key = reader.string()
                metadata[key] = reader.value(reader.unpack("<I"))

            tensors: list[TensorInfo] = []
            for _ in range(tensor_count):
                name = reader.string()
                rank = reader.unpack("<I")
                if rank > 16:
                    raise GGUFError(
                        f"tensor {name!r} has implausible rank {rank}"
                    )
                shape = tuple(reader.unpack("<Q") for _ in range(rank))
                tensors.append(
                    TensorInfo(
                        name=name,
                        shape=shape,
                        type_id=reader.unpack("<I"),
                        offset=reader.unpack("<Q"),
                    )
                )
    except OSError as error:
        raise GGUFError(f"cannot read {model_path}: {error}") from error
    return GGUFModel(model_path, version, metadata, tuple(tensors))


MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def default_ollama_models_directory() -> Path:
    configured = os.environ.get("OLLAMA_MODELS")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".ollama" / "models"
    )


def _ollama_reference_parts(
    reference: str,
) -> tuple[str, tuple[str, ...], str]:
    slash = reference.rfind("/")
    colon = reference.rfind(":")
    if colon > slash:
        repository_text, tag = reference[:colon], reference[colon + 1 :]
    else:
        repository_text, tag = reference, "latest"

    parts = repository_text.split("/")
    if not repository_text or not tag or any(
        part in {"", ".", ".."} for part in parts
    ):
        raise GGUFError(f"invalid Ollama model reference {reference!r}")
    if len(parts) > 1 and (
        "." in parts[0] or ":" in parts[0] or parts[0] == "localhost"
    ):
        registry = parts.pop(0)
    else:
        registry = "registry.ollama.ai"
    if len(parts) == 1:
        parts.insert(0, "library")
    if tag in {".", ".."} or "/" in tag or "\\" in tag:
        raise GGUFError(f"invalid Ollama model tag {tag!r}")
    return registry, tuple(parts), tag


def resolve_ollama_model(
    reference: str, models_directory: str | Path | None = None
) -> Path:
    """Resolve a model name to the single local GGUF model-layer blob."""
    models_root = (
        Path(models_directory).expanduser()
        if models_directory is not None
        else default_ollama_models_directory()
    )
    registry, repository, tag = _ollama_reference_parts(reference)
    manifest_path = models_root / "manifests" / registry
    for component in repository:
        manifest_path /= component
    manifest_path /= tag

    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except FileNotFoundError as error:
        raise GGUFError(
            f"Ollama model manifest not found: {manifest_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise GGUFError(
            f"cannot read Ollama manifest {manifest_path}: {error}"
        ) from error

    layers = manifest.get("layers") if isinstance(manifest, dict) else None
    if not isinstance(layers, list):
        raise GGUFError(f"Ollama manifest {manifest_path} has no layers array")

    candidates: list[Path] = []
    for layer in layers:
        if (
            not isinstance(layer, dict)
            or layer.get("mediaType") != MODEL_MEDIA_TYPE
        ):
            continue
        digest = layer.get("digest")
        if (
            not isinstance(digest, str)
            or DIGEST_PATTERN.fullmatch(digest) is None
        ):
            raise GGUFError(
                f"Ollama model layer has invalid digest {digest!r}"
            )
        blob = models_root / "blobs" / digest.replace(":", "-", 1)
        try:
            with blob.open("rb") as source:
                magic = source.read(4)
        except OSError as error:
            raise GGUFError(f"cannot read Ollama model blob {blob}: {error}")
        if magic == b"GGUF":
            candidates.append(blob)

    if not candidates:
        raise GGUFError(
            f"Ollama model {reference!r} has no readable GGUF model layer"
        )
    if len(candidates) > 1:
        raise GGUFError(
            f"Ollama model {reference!r} has {len(candidates)} GGUF layers; "
            "split-model visualization is not yet supported"
        )
    return candidates[0]
