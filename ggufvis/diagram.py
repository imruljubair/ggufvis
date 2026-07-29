"""Build the complete matrix layout from a normalized model configuration.

The layout is renderer-independent: this module calculates dimensions,
positions, panel bounds, and operation membership, but emits no terminal text.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import ModelConfig


# Horizontal geometry. A model with an unusually wide vocabulary label may
# shift all matrices right; ``Diagram.block_width`` records the final width.
BASE_BLOCK_WIDTH = 46
RIGHT_MATRIX_X = 24
PAIR_GAP = 6
LEVEL_ONE_X = 2
LEVEL_ONE_WIDTH = 42
LEVEL_TWO_X = 4
LEVEL_TWO_WIDTH = 38

# Panel and matrix colors are data so the renderer can remain generic.
PEACH = (255, 226, 212)
SKY = (186, 225, 241)
MLP_BACKGROUND = (232, 226, 248)
BLOCK_BACKGROUND = (235, 235, 225)
EMBEDDING_BACKGROUND = (224, 226, 230)


@dataclass(frozen=True)
class Matrix:
    key: str
    label: str
    x: int
    y: int
    columns: str
    rows: str
    panel_background: tuple[int, int, int]


@dataclass(frozen=True)
class Operation:
    """One interactive step and the matrices highlighted by that step."""

    result: str
    row: int
    matrices: frozenset[str]


@dataclass(frozen=True)
class Panels:
    embedding_bottom: int
    block_top: int
    block_bottom: int
    group_top: int
    group_bottom: int
    head_top: int
    head_bottom: int
    mlp_top: int
    mlp_bottom: int
    final_top: int
    canvas_height: int


@dataclass(frozen=True)
class Diagram:
    config: ModelConfig
    matrices: tuple[Matrix, ...]
    operations: tuple[Operation, ...]
    panels: Panels
    dimension_widths: dict[str, int]
    dimension_heights: dict[str, int]
    block_width: int
    horizontal_shift: int

    def width_for(self, dimension: str) -> int:
        return self.dimension_widths[dimension]

    def height_for(self, dimension: str) -> int:
        return self.dimension_heights[dimension]

    def matrix(self, key: str) -> Matrix:
        return next(matrix for matrix in self.matrices if matrix.key == key)


LEARNED_WEIGHTS = frozenset(
    {
        "token_embedding",
        "gamma",
        "wk",
        "gamma_k",
        "wv",
        "wq",
        "gamma_q",
        "wo",
        "gamma_post",
        "wg",
        "wu",
        "wd",
        "gamma_final",
        "wlm",
    }
)


def _dimension_encodings(
    config: ModelConfig,
) -> tuple[dict[str, int], dict[str, int]]:
    """Create ordinal terminal geometry from actual model dimensions.

    Literal proportional scaling is unusable when dimensions span 1 to more
    than 100,000. Instead, unique structural dimensions are sorted and assigned
    monotonically increasing sizes. Sequence and vocabulary intentionally share
    one height in final v1.
    """
    structural = sorted(
        {
            config.key_head_dim,
            config.value_head_dim,
            config.hidden_size,
            config.merged_value_width,
            config.ffn_size,
        }
    )
    labels = ["1", *(str(value) for value in structural), "seq"]
    vocab = str(config.vocab_size)
    if vocab not in labels:
        labels.append(vocab)

    widths: dict[str, int] = {}
    heights: dict[str, int] = {}
    for rank, label in enumerate(labels):
        if label == "1":
            widths[label] = 5
            heights[label] = 3
        else:
            widths[label] = max(len(label) + 2, min(16, 7 + 2 * rank))
            heights[label] = 3 + rank

    # A merged-head box needs room for first head, ellipsis, and last head.
    merged = str(config.merged_value_width)
    heights[merged] = max(5, heights[merged])

    # The vocabulary is the largest logical axis, but matching seq prevents the
    # embedding and LM-head panels from becoming unnecessarily tall.
    heights[vocab] = heights["seq"]
    return widths, heights


def build_diagram(config: ModelConfig) -> Diagram:
    """Calculate the complete final-v1 model diagram."""
    widths, heights = _dimension_encodings(config)
    hidden = str(config.hidden_size)
    ffn = str(config.ffn_size)
    key_dim = str(config.key_head_dim)
    value_dim = str(config.value_head_dim)
    merged = str(config.merged_value_width)
    vocab = str(config.vocab_size)

    def height(dimension: str) -> int:
        return heights[dimension]

    # Reserve enough left-side room for E's vocabulary label. The same shift is
    # applied to every later matrix, preserving column alignment.
    embedding_base_x = RIGHT_MATRIX_X - PAIR_GAP - widths[vocab]
    horizontal_shift = max(0, len(hidden) - embedding_base_x)
    right_x = RIGHT_MATRIX_X + horizontal_shift
    block_width = BASE_BLOCK_WIDTH + 2 * horizontal_shift

    def left_x(columns: str) -> int:
        return right_x - PAIR_GAP - widths[columns]

    # Embedding: the conceptual one-hot T is shown to make E × T a visible
    # matrix multiplication. Actual inference performs a token-ID lookup.
    token_y = 1
    embedding_result_y = token_y + height(vocab)
    embedding_bottom = embedding_result_y + height(hidden) + 1
    block_top = embedding_bottom

    # Transformer block rows. Optional Q/K normalization adds real matrix rows
    # before RoPE rather than being hidden in an annotation.
    y_x = block_top + 1
    y_norm = y_x + height(hidden)
    y_wk = y_norm + height(hidden) + 1
    y_k_norm = y_wk + height(key_dim)
    y_k_rope = y_k_norm + (height(key_dim) if config.qk_norm else 0)
    y_wv = y_k_rope + height(key_dim)
    y_wq = y_wv + height(value_dim) + 1
    y_q_norm = y_wq + height(key_dim)
    y_q_rope = y_q_norm + (height(key_dim) if config.qk_norm else 0)
    y_attention = y_q_rope + height(key_dim)
    y_softmax = y_attention + height("seq")
    y_head = y_softmax + height("seq")

    attention_background = PEACH
    query_background = PEACH if config.attention_kind == "MHA" else SKY

    matrices: list[Matrix] = [
        Matrix(
            "token_embedding",
            "E",
            embedding_base_x + horizontal_shift,
            embedding_result_y,
            vocab,
            hidden,
            EMBEDDING_BACKGROUND,
        ),
        Matrix(
            "token_one_hot",
            "T",
            right_x,
            token_y,
            "seq",
            vocab,
            EMBEDDING_BACKGROUND,
        ),
        Matrix(
            "embedding_x",
            "X",
            right_x,
            embedding_result_y,
            "seq",
            hidden,
            EMBEDDING_BACKGROUND,
        ),
        Matrix("x", "X", right_x, y_x, "seq", hidden, BLOCK_BACKGROUND),
        Matrix(
            "gamma", "γ", left_x("1"), y_norm, "1", hidden, BLOCK_BACKGROUND
        ),
        Matrix(
            "x_norm", "X′", right_x, y_norm, "seq", hidden, BLOCK_BACKGROUND
        ),
        Matrix(
            "wk",
            "Wk",
            left_x(hidden),
            y_wk,
            hidden,
            key_dim,
            attention_background,
        ),
        Matrix(
            "k", "K", right_x, y_wk, "seq", key_dim, attention_background
        ),
    ]
    if config.qk_norm:
        matrices.extend(
            [
                Matrix(
                    "gamma_k",
                    "γk",
                    left_x("1"),
                    y_k_norm,
                    "1",
                    key_dim,
                    attention_background,
                ),
                Matrix(
                    "k_norm",
                    "K′",
                    right_x,
                    y_k_norm,
                    "seq",
                    key_dim,
                    attention_background,
                ),
            ]
        )
    matrices.extend(
        [
            Matrix(
                "k_rope",
                "Kᵣ",
                right_x,
                y_k_rope,
                "seq",
                key_dim,
                attention_background,
            ),
            Matrix(
                "wv",
                "Wv",
                left_x(hidden),
                y_wv,
                hidden,
                value_dim,
                attention_background,
            ),
            Matrix(
                "v_projection",
                "V",
                right_x,
                y_wv,
                "seq",
                value_dim,
                attention_background,
            ),
            Matrix(
                "wq",
                "Wq",
                left_x(hidden),
                y_wq,
                hidden,
                key_dim,
                query_background,
            ),
            Matrix(
                "q", "Q", right_x, y_wq, "seq", key_dim, query_background
            ),
        ]
    )
    if config.qk_norm:
        matrices.extend(
            [
                Matrix(
                    "gamma_q",
                    "γq",
                    left_x("1"),
                    y_q_norm,
                    "1",
                    key_dim,
                    query_background,
                ),
                Matrix(
                    "q_norm",
                    "Q′",
                    right_x,
                    y_q_norm,
                    "seq",
                    key_dim,
                    query_background,
                ),
            ]
        )
    matrices.extend(
        [
            Matrix(
                "q_rope",
                "Qᵣ",
                right_x,
                y_q_rope,
                "seq",
                key_dim,
                query_background,
            ),
            Matrix(
                "kt_rope",
                "Kᵣᵀ",
                left_x(key_dim),
                y_attention,
                key_dim,
                "seq",
                query_background,
            ),
            Matrix(
                "a",
                "A",
                right_x,
                y_attention,
                "seq",
                "seq",
                query_background,
            ),
            Matrix(
                "a_softmax",
                "A′",
                right_x,
                y_softmax,
                "seq",
                "seq",
                query_background,
            ),
            Matrix(
                "v_attention",
                "V",
                (6 if config.attention_kind == "GQA" else 4)
                + horizontal_shift,
                y_head,
                "seq",
                value_dim,
                query_background,
            ),
            Matrix(
                "h",
                "Hᵢⱼ" if config.attention_kind == "GQA" else "Hᵢ",
                right_x,
                y_head,
                "seq",
                value_dim,
                query_background,
            ),
        ]
    )

    head_keys = {
        "wq",
        "q",
        "q_rope",
        "kt_rope",
        "a",
        "a_softmax",
        "v_attention",
        "h",
    }
    if config.qk_norm:
        head_keys |= {"gamma_q", "q_norm"}
    group_keys = head_keys | {
        "wk",
        "k",
        "k_rope",
        "wv",
        "v_projection",
    }
    if config.qk_norm:
        group_keys |= {"gamma_k", "k_norm"}

    matrix_by_key = {matrix.key: matrix for matrix in matrices}
    group_top = min(matrix_by_key[key].y for key in group_keys) - 1
    head_top = min(matrix_by_key[key].y for key in head_keys) - 1
    attention_content_bottom = max(
        matrix_by_key[key].y + height(matrix_by_key[key].rows) - 1
        for key in group_keys
    )
    if config.attention_kind == "GQA":
        head_bottom = attention_content_bottom + 2
        group_bottom = attention_content_bottom + 3
    else:
        group_bottom = attention_content_bottom + 2
        head_bottom = group_bottom

    # Everything below attention is positioned from the calculated panel
    # borders, so nested borders cannot overlap the merged-head matrix.
    y_merge = group_bottom
    y_wo = y_merge + height(merged)
    y_residual = y_wo + height(hidden)
    y_post_norm = y_residual + height(hidden)
    y_wg = y_post_norm + height(hidden) + 2
    y_wu = y_wg + height(ffn)
    y_product = y_wu + height(ffn)
    y_wd = y_product + height(ffn)
    y_block_residual = y_wd + height(hidden)

    matrices.extend(
        [
            Matrix(
                "h_merge",
                "Hmerge",
                right_x,
                y_merge,
                "seq",
                merged,
                BLOCK_BACKGROUND,
            ),
            Matrix(
                "wo",
                "Wₒ",
                left_x(merged),
                y_wo,
                merged,
                hidden,
                BLOCK_BACKGROUND,
            ),
            Matrix("y", "Y", right_x, y_wo, "seq", hidden, BLOCK_BACKGROUND),
            Matrix(
                "x_residual",
                "X",
                left_x("seq"),
                y_residual,
                "seq",
                hidden,
                BLOCK_BACKGROUND,
            ),
            Matrix(
                "r", "R", right_x, y_residual, "seq", hidden, BLOCK_BACKGROUND
            ),
            Matrix(
                "gamma_post",
                "γ′",
                left_x("1"),
                y_post_norm,
                "1",
                hidden,
                BLOCK_BACKGROUND,
            ),
            Matrix(
                "r_norm",
                "R′",
                right_x,
                y_post_norm,
                "seq",
                hidden,
                BLOCK_BACKGROUND,
            ),
            Matrix(
                "wg",
                "Wg",
                left_x(hidden),
                y_wg,
                hidden,
                ffn,
                MLP_BACKGROUND,
            ),
            Matrix("g", "G", right_x, y_wg, "seq", ffn, MLP_BACKGROUND),
            Matrix(
                "wu",
                "Wu",
                left_x(hidden),
                y_wu,
                hidden,
                ffn,
                MLP_BACKGROUND,
            ),
            Matrix("u", "U", right_x, y_wu, "seq", ffn, MLP_BACKGROUND),
            Matrix(
                "p", "P", right_x, y_product, "seq", ffn, MLP_BACKGROUND
            ),
            Matrix(
                "wd",
                "Wd",
                left_x(ffn),
                y_wd,
                ffn,
                hidden,
                MLP_BACKGROUND,
            ),
            Matrix("m", "M", right_x, y_wd, "seq", hidden, MLP_BACKGROUND),
            Matrix(
                "r_final",
                "R",
                4 + horizontal_shift,
                y_block_residual,
                "seq",
                hidden,
                MLP_BACKGROUND,
            ),
            Matrix(
                "x_out",
                "Xout",
                right_x,
                y_block_residual,
                "seq",
                hidden,
                MLP_BACKGROUND,
            ),
        ]
    )

    mlp_top = y_wg - 1
    mlp_content_end = y_block_residual + height(hidden)
    mlp_bottom = mlp_content_end + 1
    block_bottom = mlp_bottom + 1
    final_top = block_bottom
    y_final_norm = final_top + 1
    y_lm = y_final_norm + height(hidden)
    canvas_height = y_lm + height(vocab) + 1

    matrices.extend(
        [
            Matrix(
                "gamma_final",
                "γf",
                left_x("1"),
                y_final_norm,
                "1",
                hidden,
                PEACH,
            ),
            Matrix("z", "Z", right_x, y_final_norm, "seq", hidden, PEACH),
            Matrix(
                "wlm",
                "Eᵀ" if config.output_is_tied else "Wlm",
                left_x(hidden),
                y_lm,
                hidden,
                vocab,
                PEACH,
            ),
            Matrix("logits", "Logits", right_x, y_lm, "seq", vocab, PEACH),
        ]
    )

    operations: list[Operation] = [
        Operation(
            "embedding_x",
            embedding_result_y + 2,
            frozenset({"token_embedding", "token_one_hot", "embedding_x"}),
        ),
        Operation(
            "x_norm",
            y_norm + 2,
            frozenset({"x", "gamma", "x_norm"}),
        ),
        Operation("k", y_wk + 1, frozenset({"x_norm", "wk", "k"})),
    ]
    if config.qk_norm:
        operations.append(
            Operation(
                "k_norm",
                y_k_norm + 1,
                frozenset({"k", "gamma_k", "k_norm"}),
            )
        )
    operations.extend(
        [
            Operation(
                "k_rope",
                y_k_rope + 1,
                frozenset(
                    {
                        "k_norm" if config.qk_norm else "k",
                        "k_rope",
                    }
                ),
            ),
            Operation(
                "v_projection",
                y_wv + 1,
                frozenset({"x_norm", "wv", "v_projection"}),
            ),
            Operation("q", y_wq + 1, frozenset({"x_norm", "wq", "q"})),
        ]
    )
    if config.qk_norm:
        operations.append(
            Operation(
                "q_norm",
                y_q_norm + 1,
                frozenset({"q", "gamma_q", "q_norm"}),
            )
        )
    operations.extend(
        [
            Operation(
                "q_rope",
                y_q_rope + 1,
                frozenset(
                    {
                        "q_norm" if config.qk_norm else "q",
                        "q_rope",
                    }
                ),
            ),
            Operation(
                "a",
                y_attention + 2,
                frozenset({"kt_rope", "q_rope", "a"}),
            ),
            Operation(
                "a_softmax",
                y_softmax + 2,
                frozenset({"a", "a_softmax"}),
            ),
            Operation(
                "h",
                y_head + 1,
                frozenset({"v_attention", "a_softmax", "h"}),
            ),
            Operation(
                "h_merge",
                y_merge + 2,
                frozenset({"h", "h_merge"}),
            ),
            Operation(
                "y", y_wo + 2, frozenset({"wo", "h_merge", "y"})
            ),
            Operation(
                "r",
                y_residual + 2,
                frozenset({"x_residual", "y", "r"}),
            ),
            Operation(
                "r_norm",
                y_post_norm + 2,
                frozenset({"r", "gamma_post", "r_norm"}),
            ),
            Operation("g", y_wg + 2, frozenset({"r_norm", "wg", "g"})),
            Operation("u", y_wu + 2, frozenset({"r_norm", "wu", "u"})),
            Operation(
                "p", y_product + 2, frozenset({"g", "u", "p"})
            ),
            Operation("m", y_wd + 2, frozenset({"wd", "p", "m"})),
            Operation(
                "x_out",
                y_block_residual + 2,
                frozenset({"r_final", "m", "x_out"}),
            ),
            Operation(
                "z",
                y_final_norm + 2,
                frozenset({"x_out", "gamma_final", "z"}),
            ),
            Operation(
                "logits",
                y_lm + 2,
                frozenset({"wlm", "z", "logits"}),
            ),
        ]
    )

    return Diagram(
        config=config,
        matrices=tuple(matrices),
        operations=tuple(operations),
        panels=Panels(
            embedding_bottom=embedding_bottom,
            block_top=block_top,
            block_bottom=block_bottom,
            group_top=group_top,
            group_bottom=group_bottom,
            head_top=head_top,
            head_bottom=head_bottom,
            mlp_top=mlp_top,
            mlp_bottom=mlp_bottom,
            final_top=final_top,
            canvas_height=canvas_height,
        ),
        dimension_widths=widths,
        dimension_heights=heights,
        block_width=block_width,
        horizontal_shift=horizontal_shift,
    )
