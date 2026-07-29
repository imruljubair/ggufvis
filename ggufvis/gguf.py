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
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


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
    source_url: str | None = None
    remote_bytes_transferred: int | None = None
    remote_file_size: int | None = None

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

CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
REMOTE_CHUNK_SIZE = 64 * 1024
MAX_REMOTE_HEADER_BYTES = 256 * 1024 * 1024
MAX_REMOTE_REQUESTS = 10_000


def _normalize_remote_url(url: str) -> str:
    """Validate HTTP(S) and convert Hugging Face browser links to raw links."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise GGUFError("remote GGUF URL must use http:// or https://")
    path = parts.path
    if parts.hostname and parts.hostname.endswith("huggingface.co"):
        path = path.replace("/blob/", "/resolve/", 1)
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


class HTTPRangeSource:
    """A seekable, in-memory view over an HTTP byte-range resource.

    The GGUF reader mostly advances sequentially, but its metadata skipper can
    jump over large arrays. Range requests make those seeks inexpensive and
    ensure the tensor payload is never saved as a local file.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 20.0,
        chunk_size: int = REMOTE_CHUNK_SIZE,
        max_bytes: int = MAX_REMOTE_HEADER_BYTES,
    ) -> None:
        self.url = _normalize_remote_url(url)
        self.timeout = timeout
        self.chunk_size = max(4096, chunk_size)
        self.max_bytes = max_bytes
        self.position = 0
        self.cache_start = 0
        self.cache = b""
        self.bytes_transferred = 0
        self.total_size: int | None = None
        self.request_count = 0
        self.final_url = self.url

        self.headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "ggufvis/1.0",
        }

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2 and self.total_size is not None:
            position = self.total_size + offset
        else:
            raise GGUFError("unsupported remote seek operation")
        if position < 0:
            raise GGUFError("cannot seek before the start of a remote GGUF")
        self.position = position
        return self.position

    def _fetch(self, start: int, minimum_size: int) -> None:
        if self.request_count >= MAX_REMOTE_REQUESTS:
            raise GGUFError("remote GGUF required too many HTTP range requests")
        fetch_size = max(self.chunk_size, minimum_size)
        end = start + fetch_size - 1
        headers = dict(self.headers)
        headers["Range"] = f"bytes={start}-{end}"
        request = Request(self.url, headers=headers)
        self.request_count += 1
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", response.getcode())
                if status != 206:
                    raise GGUFError(
                        "remote server did not honor HTTP byte ranges; use a "
                        "direct GGUF URL whose server supports Range requests"
                    )
                content_range = response.headers.get("Content-Range", "")
                match = CONTENT_RANGE_PATTERN.fullmatch(content_range.strip())
                if match is None or int(match.group(1)) != start:
                    raise GGUFError(
                        f"invalid HTTP Content-Range {content_range!r}"
                    )
                total = match.group(3)
                if total != "*":
                    self.total_size = int(total)
                data = response.read()
                self.final_url = response.geturl()
        except HTTPError as error:
            if error.code == 416:
                data = b""
            else:
                raise GGUFError(
                    f"remote GGUF request failed with HTTP {error.code}"
                ) from error
        except URLError as error:
            raise GGUFError(f"cannot read remote GGUF: {error.reason}") from error
        except OSError as error:
            raise GGUFError(f"cannot read remote GGUF: {error}") from error

        self.bytes_transferred += len(data)
        if self.bytes_transferred > self.max_bytes:
            raise GGUFError(
                "remote GGUF metadata/tensor descriptors exceeded the "
                f"{self.max_bytes // (1024 * 1024)} MiB safety limit"
            )
        self.cache_start = start
        self.cache = data

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise GGUFError("unbounded reads are disabled for remote GGUFs")
        output = bytearray()
        while len(output) < size:
            cache_end = self.cache_start + len(self.cache)
            if not (self.cache_start <= self.position < cache_end):
                self._fetch(self.position, size - len(output))
                cache_end = self.cache_start + len(self.cache)
                if not self.cache:
                    break
            available = cache_end - self.position
            take = min(size - len(output), available)
            offset = self.position - self.cache_start
            output.extend(self.cache[offset : offset + take])
            self.position += take
        return bytes(output)


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


def _read_gguf_source(
    source,
    model_path: Path,
    *,
    source_url: str | None = None,
    remote_bytes_transferred: int | None = None,
    remote_file_size: int | None = None,
) -> GGUFModel:
    """Parse one local or remote seekable source through the same code path."""
    reader = _Reader(source)
    if reader.exact(4) != b"GGUF":
        raise GGUFError(f"{model_path} is not a GGUF file")
    version = reader.unpack("<I")
    if version not in {2, 3}:
        raise GGUFError(f"unsupported GGUF version {version}; expected 2 or 3")
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
            raise GGUFError(f"tensor {name!r} has implausible rank {rank}")
        shape = tuple(reader.unpack("<Q") for _ in range(rank))
        tensors.append(
            TensorInfo(
                name=name,
                shape=shape,
                type_id=reader.unpack("<I"),
                offset=reader.unpack("<Q"),
            )
        )
    return GGUFModel(
        model_path,
        version,
        metadata,
        tuple(tensors),
        source_url=source_url,
        remote_bytes_transferred=remote_bytes_transferred,
        remote_file_size=remote_file_size,
    )


def read_gguf(path: str | Path) -> GGUFModel:
    """Read local metadata and tensor shapes without reading tensor data."""
    model_path = Path(path)
    try:
        with model_path.open("rb") as source:
            return _read_gguf_source(source, model_path)
    except OSError as error:
        raise GGUFError(f"cannot read {model_path}: {error}") from error


def read_gguf_url(url: str) -> GGUFModel:
    """Read a remote GGUF header through bounded, in-memory range requests."""
    source = HTTPRangeSource(url)
    normalized = source.url
    filename = unquote(Path(urlsplit(normalized).path).name) or "remote.gguf"
    # Remote counters are known only after parsing, so update the immutable
    # result once the shared parser has stopped after tensor descriptors.
    model = _read_gguf_source(source, Path(filename), source_url=normalized)
    return GGUFModel(
        model.path,
        model.version,
        model.metadata,
        model.tensors,
        source_url=source.final_url,
        remote_bytes_transferred=source.bytes_transferred,
        remote_file_size=source.total_size,
    )


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
