"""Normalize supported GGUF architectures into one renderer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gguf import ArrayInfo, GGUFError, GGUFModel, TensorInfo


SUPPORTED_ARCHITECTURES = frozenset({"llama", "gemma", "qwen2", "qwen3"})


@dataclass(frozen=True)
class ModelConfig:
    """All dimensions and architecture flags required by the diagram."""

    architecture: str
    name: str
    is_deepseek_r1_distill: bool
    block_count: int
    hidden_size: int
    ffn_size: int
    query_heads: int
    kv_heads: int
    key_head_dim: int
    value_head_dim: int
    vocab_size: int
    context_length: int | None
    output_is_tied: bool
    qk_norm: bool
    qkv_bias: bool
    activation: str
    warnings: tuple[str, ...]

    @property
    def queries_per_kv(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def query_width(self) -> int:
        return self.query_heads * self.key_head_dim

    @property
    def key_width(self) -> int:
        return self.kv_heads * self.key_head_dim

    @property
    def value_width(self) -> int:
        return self.kv_heads * self.value_head_dim

    @property
    def merged_value_width(self) -> int:
        return self.query_heads * self.value_head_dim

    @property
    def attention_kind(self) -> str:
        if self.query_heads == self.kv_heads:
            return "MHA"
        if self.kv_heads == 1:
            return "MQA"
        return "GQA"

    @property
    def family_name(self) -> str:
        backbone = self.backbone_name
        if self.is_deepseek_r1_distill:
            return f"DeepSeek-R1 Distill ({backbone})"
        return backbone

    @property
    def backbone_name(self) -> str:
        """Human-readable name of the architecture that defines the graph."""
        return {
            "llama": "Llama",
            "gemma": "Gemma",
            "qwen2": "Qwen2",
            "qwen3": "Qwen3",
        }[self.architecture]

    @property
    def annotation_identity(self) -> str:
        """Identity shown on every card without changing graph dispatch."""
        if self.is_deepseek_r1_distill:
            return (
                f"DEEPSEEK-R1 DISTILL · "
                f"{self.backbone_name.upper()} BACKBONE"
            )
        return self.backbone_name.upper()


def _integer(
    metadata: dict[str, Any], key: str, *, required: bool = False
) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        value = None
    if isinstance(value, (int, float)):
        return int(value)
    if required:
        raise GGUFError(f"required metadata {key!r} is missing")
    return None


def _projection_width(tensor: TensorInfo | None, hidden: int) -> int | None:
    if tensor is None or len(tensor.shape) != 2:
        return None
    first, second = tensor.shape
    if first == hidden:
        return second
    if second == hidden:
        return first
    return None


def _other_dimension(tensor: TensorInfo | None, known: int) -> int | None:
    if tensor is None or len(tensor.shape) != 2:
        return None
    first, second = tensor.shape
    if first == known:
        return second
    if second == known:
        return first
    return None


def _matrix_matches(
    tensor: TensorInfo | None, first: int, second: int
) -> bool | None:
    if tensor is None or len(tensor.shape) != 2:
        return None
    return tensor.shape in {(first, second), (second, first)}


def _is_deepseek_r1_distill(
    metadata: dict[str, Any], architecture: str
) -> bool:
    """Recognize DeepSeek-R1 models carried by a supported dense backbone.

    Some GGUFs include ``Distill`` in ``general.name`` while newer models such
    as DeepSeek-R1-0528-Qwen3-8B do not.  A DeepSeek-R1 identity paired with a
    supported Qwen/Llama architecture is sufficient: the architecture remains
    the source of all dimensions and matrix-layout decisions.
    """
    identity_fields = (
        "general.name",
        "general.basename",
        "general.finetune",
        "general.description",
    )
    identity = " ".join(
        value
        for key in identity_fields
        if isinstance((value := metadata.get(key)), str)
    ).lower()
    normalized = "".join(
        character if character.isalnum() else " " for character in identity
    )
    words = normalized.split()
    return (
        architecture in {"llama", "qwen2", "qwen3"}
        and "deepseek" in words
        and "r1" in words
    )


def _find_vocab(model: GGUFModel, prefix: str, hidden: int) -> int:
    explicit = _integer(model.metadata, f"{prefix}.vocab_size")
    if explicit:
        return explicit
    tokens = model.metadata.get("tokenizer.ggml.tokens")
    if isinstance(tokens, ArrayInfo) and tokens.count:
        return tokens.count
    if isinstance(tokens, tuple) and tokens:
        return len(tokens)
    for name in ("token_embd.weight", "output.weight"):
        value = _other_dimension(model.tensor(name), hidden)
        if value:
            return value
    raise GGUFError(
        "cannot determine vocabulary size from metadata or tensor shapes"
    )


def _validate_optional_pair(
    model: GGUFModel,
    first_name: str,
    second_name: str,
    expected_shape: tuple[int, ...],
    description: str,
) -> bool:
    first = model.tensor(first_name)
    second = model.tensor(second_name)
    if (first is None) != (second is None):
        raise GGUFError(f"GGUF contains an incomplete {description} tensor pair")
    for tensor in (first, second):
        if tensor is not None and tensor.shape != expected_shape:
            raise GGUFError(
                f"{tensor.name} has shape {tensor.shape}, expected "
                f"{expected_shape}"
            )
    return first is not None


def config_from_gguf(model: GGUFModel) -> ModelConfig:
    """Build a dimension-driven dense model configuration."""
    metadata = model.metadata
    architecture = metadata.get("general.architecture")
    if architecture in {"qwen2moe", "qwen3moe"}:
        raise GGUFError(
            f"{architecture} is a Qwen MoE architecture; final v1 supports "
            "dense Qwen2/Qwen2.5/Qwen3"
        )
    if isinstance(architecture, str) and architecture.startswith("deepseek"):
        raise GGUFError(
            f"{architecture} is a native DeepSeek architecture; final v1 "
            "supports DeepSeek-R1 Distill models only when their GGUF "
            "backbone architecture is qwen2, qwen3, or llama"
        )
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise GGUFError(
            f"unsupported architecture {architecture!r}; final v1 supports "
            "dense Llama, original Gemma, Qwen2/Qwen2.5, and Qwen3"
        )
    prefix = str(architecture)

    hidden = _integer(metadata, f"{prefix}.embedding_length", required=True)
    ffn = _integer(metadata, f"{prefix}.feed_forward_length", required=True)
    blocks = _integer(metadata, f"{prefix}.block_count", required=True)
    query_heads = _integer(
        metadata, f"{prefix}.attention.head_count", required=True
    )
    kv_heads = (
        _integer(metadata, f"{prefix}.attention.head_count_kv") or query_heads
    )
    assert hidden is not None and ffn is not None and blocks is not None
    assert query_heads is not None and kv_heads is not None

    if min(hidden, ffn, blocks, query_heads, kv_heads) <= 0:
        raise GGUFError("model dimensions and head counts must be positive")
    if query_heads % kv_heads:
        raise GGUFError(
            f"query heads {query_heads} are not divisible by KV heads {kv_heads}"
        )

    experts = _integer(metadata, f"{prefix}.expert_count") or 0
    if experts:
        raise GGUFError(
            f"{architecture} declares {experts} experts; final v1 does not "
            "yet visualize MoE blocks"
        )

    q_width = _projection_width(model.tensor("blk.0.attn_q.weight"), hidden)
    v_width = _projection_width(model.tensor("blk.0.attn_v.weight"), hidden)
    key_head_dim = _integer(metadata, f"{prefix}.attention.key_length")
    if key_head_dim is None and q_width and q_width % query_heads == 0:
        key_head_dim = q_width // query_heads
    if key_head_dim is None and hidden % query_heads == 0:
        key_head_dim = hidden // query_heads
    if key_head_dim is None:
        raise GGUFError("cannot derive the attention key head dimension")

    value_head_dim = _integer(metadata, f"{prefix}.attention.value_length")
    if value_head_dim is None and v_width and v_width % kv_heads == 0:
        value_head_dim = v_width // kv_heads
    if value_head_dim is None:
        value_head_dim = key_head_dim

    qk_norm = _validate_optional_pair(
        model,
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        (key_head_dim,),
        "Q/K normalization",
    )
    bias_names = (
        "blk.0.attn_q.bias",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.bias",
    )
    present_biases = tuple(model.tensor(name) is not None for name in bias_names)
    if any(present_biases) and not all(present_biases):
        raise GGUFError("GGUF contains an incomplete Q/K/V projection bias set")
    qkv_bias = all(present_biases)

    vocab = _find_vocab(model, prefix, hidden)
    context = _integer(metadata, f"{prefix}.context_length")
    name_value = metadata.get("general.name")
    name = name_value if isinstance(name_value, str) else model.path.stem

    expected = {
        "token_embd.weight": (hidden, vocab),
        "blk.0.attn_q.weight": (hidden, query_heads * key_head_dim),
        "blk.0.attn_k.weight": (hidden, kv_heads * key_head_dim),
        "blk.0.attn_v.weight": (hidden, kv_heads * value_head_dim),
        "blk.0.attn_output.weight": (
            query_heads * value_head_dim,
            hidden,
        ),
        "blk.0.ffn_gate.weight": (hidden, ffn),
        "blk.0.ffn_up.weight": (hidden, ffn),
        "blk.0.ffn_down.weight": (ffn, hidden),
    }
    warnings: list[str] = []
    for tensor_name, dimensions in expected.items():
        tensor = model.tensor(tensor_name)
        if _matrix_matches(tensor, *dimensions) is False:
            warnings.append(
                f"{tensor_name} has GGUF shape {tensor.shape}, expected axes "
                f"{dimensions} in either storage order"
            )

    layer_indices = {
        int(parts[1])
        for tensor in model.tensors
        if len(parts := tensor.name.split(".")) > 2
        and parts[0] == "blk"
        and parts[1].isdigit()
    }
    if layer_indices and max(layer_indices) + 1 != blocks:
        warnings.append(
            f"metadata declares {blocks} blocks, tensor names reach "
            f"block {max(layer_indices)}"
        )

    return ModelConfig(
        architecture=prefix,
        name=name,
        is_deepseek_r1_distill=_is_deepseek_r1_distill(metadata, prefix),
        block_count=blocks,
        hidden_size=hidden,
        ffn_size=ffn,
        query_heads=query_heads,
        kv_heads=kv_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        vocab_size=vocab,
        context_length=context,
        output_is_tied=model.tensor("output.weight") is None,
        qk_norm=qk_norm,
        qkv_bias=qkv_bias,
        activation="GELU" if architecture == "gemma" else "SiLU",
        warnings=tuple(warnings),
    )
