"""Generate the final-v1 structured annotation cards."""

from __future__ import annotations

from .diagram import Diagram


SECTION_BY_RESULT = {
    "embedding_x": "EMBEDDING",
    "x_norm": "ATTENTION",
    "k": "ATTENTION",
    "k_norm": "ATTENTION",
    "k_rope": "ATTENTION",
    "v_projection": "ATTENTION",
    "q": "ATTENTION",
    "q_norm": "ATTENTION",
    "q_rope": "ATTENTION",
    "a": "ATTENTION",
    "a_softmax": "ATTENTION",
    "h": "ATTENTION",
    "h_merge": "ATTENTION",
    "y": "ATTENTION",
    "r": "ATTENTION",
    "r_norm": "MLP",
    "g": "MLP",
    "u": "MLP",
    "p": "MLP",
    "m": "MLP",
    "x_out": "MLP",
    "z": "FINAL",
    "logits": "FINAL",
}

OPERATION_BY_RESULT = {
    "embedding_x": "Token embedding",
    "x_norm": "Attention RMSNorm",
    "k": "K projection",
    "k_norm": "K per-head RMSNorm",
    "k_rope": "K rotary position encoding",
    "v_projection": "V projection",
    "q": "Q projection",
    "q_norm": "Q per-head RMSNorm",
    "q_rope": "Q rotary position encoding",
    "a": "Attention scores",
    "a_softmax": "Causal softmax",
    "h": "Weighted value aggregation",
    "h_merge": "Head concatenation",
    "y": "Attention output projection",
    "r": "Attention residual",
    "r_norm": "MLP RMSNorm",
    "g": "Gate projection",
    "u": "Up projection",
    "p": "Gated activation",
    "m": "Down projection",
    "x_out": "MLP residual",
    "z": "Final RMSNorm",
    "logits": "Language-model head",
}


def _shape(diagram: Diagram, key: str) -> str:
    matrix = diagram.matrix(key)
    # Equations use mathematical row×column order. GGUF may store these axes in
    # the reverse order; tensor validation accepts either storage orientation.
    return f"{matrix.label}[{matrix.rows}×{matrix.columns}]"


def equation_for(diagram: Diagram, result: str) -> str:
    """Return a shape-aware conceptual equation for one result matrix."""
    shape = lambda key: _shape(diagram, key)
    config = diagram.config
    bias = config.qkv_bias
    k_input = "k_norm" if config.qk_norm else "k"
    q_input = "q_norm" if config.qk_norm else "q"
    equations = {
        "embedding_x": (
            f"{shape('embedding_x')} = {shape('token_embedding')} × "
            f"{shape('token_one_hot')}"
        ),
        "x_norm": (
            f"{shape('x_norm')} = RMSNorm({shape('x')}; {shape('gamma')})"
        ),
        "k": (
            f"{shape('k')} = {shape('wk')} × {shape('x_norm')}"
            + (f" + bk[{config.key_head_dim}×1]" if bias else "")
        ),
        "k_rope": f"{shape('k_rope')} = RoPE({shape(k_input)})",
        "v_projection": (
            f"{shape('v_projection')} = {shape('wv')} × {shape('x_norm')}"
            + (f" + bv[{config.value_head_dim}×1]" if bias else "")
        ),
        "q": (
            f"{shape('q')} = {shape('wq')} × {shape('x_norm')}"
            + (f" + bq[{config.key_head_dim}×1]" if bias else "")
        ),
        "q_rope": f"{shape('q_rope')} = RoPE({shape(q_input)})",
        "a": f"{shape('a')} = {shape('kt_rope')} × {shape('q_rope')}",
        "a_softmax": (
            f"{shape('a_softmax')} = softmax({shape('a')}/"
            f"√{config.key_head_dim} + Mᶜ)"
        ),
        "h": (
            f"{shape('h')} = {shape('v_attention')} × "
            f"{shape('a_softmax')}"
        ),
        "h_merge": (
            f"{shape('h_merge')} = Concat("
            f"{config.query_heads} heads × {config.value_head_dim})"
        ),
        "y": f"{shape('y')} = {shape('wo')} × {shape('h_merge')}",
        "r": f"{shape('r')} = {shape('x_residual')} + {shape('y')}",
        "r_norm": (
            f"{shape('r_norm')} = RMSNorm({shape('r')}; "
            f"{shape('gamma_post')})"
        ),
        "g": f"{shape('g')} = {shape('wg')} × {shape('r_norm')}",
        "u": f"{shape('u')} = {shape('wu')} × {shape('r_norm')}",
        "p": (
            f"{shape('p')} = {config.activation}({shape('g')}) "
            f"⊙ {shape('u')}"
        ),
        "m": f"{shape('m')} = {shape('wd')} × {shape('p')}",
        "x_out": f"{shape('x_out')} = {shape('r_final')} + {shape('m')}",
        "z": (
            f"{shape('z')} = RMSNorm({shape('x_out')}; "
            f"{shape('gamma_final')})"
        ),
        "logits": f"{shape('logits')} = {shape('wlm')} × {shape('z')}",
    }
    if config.qk_norm:
        equations["k_norm"] = (
            f"{shape('k_norm')} = RMSNorm({shape('k')}; {shape('gamma_k')})"
        )
        equations["q_norm"] = (
            f"{shape('q_norm')} = RMSNorm({shape('q')}; {shape('gamma_q')})"
        )
    return equations[result]


def tensor_source_for(diagram: Diagram, result: str) -> str:
    """Return the canonical GGUF tensor source, never a weight value."""
    config = diagram.config
    bias = config.qkv_bias
    sources = {
        "embedding_x": "token_embd.weight",
        "x_norm": "blk.b.attn_norm.weight",
        "k": (
            "blk.b.attn_k.weight + blk.b.attn_k.bias"
            if bias
            else "blk.b.attn_k.weight"
        ),
        "k_norm": "blk.b.attn_k_norm.weight",
        "k_rope": "none (parameter-free)",
        "v_projection": (
            "blk.b.attn_v.weight + blk.b.attn_v.bias"
            if bias
            else "blk.b.attn_v.weight"
        ),
        "q": (
            "blk.b.attn_q.weight + blk.b.attn_q.bias"
            if bias
            else "blk.b.attn_q.weight"
        ),
        "q_norm": "blk.b.attn_q_norm.weight",
        "q_rope": "none (parameter-free)",
        "a": "none (parameter-free)",
        "a_softmax": "none (parameter-free)",
        "h": "none (parameter-free)",
        "h_merge": "none (parameter-free)",
        "y": "blk.b.attn_output.weight",
        "r": "none (parameter-free)",
        "r_norm": "blk.b.ffn_norm.weight",
        "g": "blk.b.ffn_gate.weight",
        "u": "blk.b.ffn_up.weight",
        "p": "none (parameter-free)",
        "m": "blk.b.ffn_down.weight",
        "x_out": "none (parameter-free)",
        "z": "output_norm.weight",
        "logits": (
            "token_embd.weight (transposed, tied)"
            if config.output_is_tied
            else "output.weight"
        ),
    }
    return sources[result]


def card_lines(diagram: Diagram, step: int) -> tuple[str, str, str, str]:
    """Return title, operation, equation, and tensor-source rows."""
    result = diagram.operations[step].result
    return (
        f"[{step + 1:02d}/{len(diagram.operations):02d}] "
        f"{SECTION_BY_RESULT[result]}",
        f"Operation: {OPERATION_BY_RESULT[result]}",
        f"Equation: {equation_for(diagram, result)}",
        f"Tensor source: {tensor_source_for(diagram, result)}",
    )


def maximum_card_width(diagram: Diagram) -> int:
    """Pre-calculate a stable card width for every navigation step."""
    longest = max(
        len(line)
        for step in range(len(diagram.operations))
        for line in card_lines(diagram, step)
    )
    return longest + 4
