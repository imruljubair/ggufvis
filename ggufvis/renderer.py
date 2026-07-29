"""Render a final-v1 Diagram onto the local terminal Canvas."""

from __future__ import annotations

from .annotations import card_lines, maximum_card_width
from .diagram import (
    BLOCK_BACKGROUND,
    EMBEDDING_BACKGROUND,
    LEARNED_WEIGHTS,
    LEVEL_ONE_WIDTH,
    LEVEL_ONE_X,
    LEVEL_TWO_WIDTH,
    LEVEL_TWO_X,
    MLP_BACKGROUND,
    PEACH,
    SKY,
    Diagram,
    Matrix,
)
from .terminal import Canvas, Color, put


BLOCK_BORDER = (105, 90, 45)
GROUP_BORDER = (185, 65, 45)
HEAD_BORDER = (30, 95, 155)
EMBEDDING_BORDER = (95, 100, 110)
MLP_BORDER = (145, 40, 125)
FINAL_BORDER: Color = "bright_red"

ACTIVE_BORDER = (0, 0, 0)
INACTIVE_BORDER = (135, 135, 135)
INACTIVE_TEXT = (145, 145, 145)
ACTIVE_TEXT = (20, 20, 20)
RESULT_FILL = (205, 65, 75)
RESULT_BORDER = (120, 25, 35)
RESULT_TEXT = (255, 255, 255)

# Final-v1 active fills are slightly darker than the original palette.
ACTIVE_INPUT_FILL = (193, 228, 193)
ACTIVE_WEIGHT_FILL = (243, 210, 168)
INACTIVE_DARKENING = 10

CARD_BACKGROUND = (30, 27, 34)
CARD_BORDER = (235, 70, 205)
CARD_LABEL = (190, 155, 205)
CARD_TEXT = (235, 235, 235)
CARD_HEIGHT = 5


def _inactive_fill(
    background: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(
        max(0, channel - INACTIVE_DARKENING) for channel in background
    )


def _panel(
    canvas: Canvas,
    x: int,
    y: int,
    width: int,
    height: int,
    background: tuple[int, int, int],
    title: str,
    border: Color,
) -> None:
    """Draw a filled single-line panel with a title in its top border."""
    canvas.fill_rect(x, y, width, height, background)
    canvas.fancy_box(x, y, width, height, "single", border)
    # Reapply backgrounds to border cells so the ANSI render has no gaps.
    for column in range(x, x + width):
        canvas.background_colors[y][column] = background
        canvas.background_colors[y + height - 1][column] = background
    for row in range(y, y + height):
        canvas.background_colors[row][x] = background
        canvas.background_colors[row][x + width - 1] = background
    put(
        canvas,
        x + 2,
        y,
        f" {title} ",
        border,
        background,
        bold=True,
    )


def _end_label(
    canvas: Canvas,
    *,
    x: int,
    bottom: int,
    width: int,
    name: str,
    color: Color,
    background: tuple[int, int, int],
) -> None:
    """Right-align ``name end`` inside an existing lower panel border."""
    available = max(0, width - 4)
    text = f" {name} end "
    if len(text) > available:
        text = f" {name[:max(1, available - 6)]} end "
    start = x + width - 1 - len(text)
    put(canvas, start, bottom, text, color, background, bold=True)


def _draw_matrix(
    canvas: Canvas,
    diagram: Diagram,
    matrix: Matrix,
    active: bool,
    result: bool,
) -> None:
    width = diagram.width_for(matrix.columns)
    height = diagram.height_for(matrix.rows)
    weight = matrix.key in LEARNED_WEIGHTS
    fill = (
        RESULT_FILL
        if result
        else ACTIVE_WEIGHT_FILL
        if active and weight
        else ACTIVE_INPUT_FILL
        if active
        else _inactive_fill(matrix.panel_background)
    )
    interior = (
        RESULT_TEXT
        if result
        else ACTIVE_TEXT
        if active
        else INACTIVE_TEXT
    )
    exterior = ACTIVE_TEXT if active else INACTIVE_TEXT
    border = (
        RESULT_BORDER
        if result
        else ACTIVE_BORDER
        if active
        else INACTIVE_BORDER
    )
    style = "double" if active else "heavy"

    canvas.fill_rect(matrix.x, matrix.y, width, height, fill)
    canvas.fancy_box(matrix.x, matrix.y, width, height, style, border)
    for column in range(matrix.x, matrix.x + width):
        canvas.background_colors[matrix.y][column] = fill
        canvas.background_colors[matrix.y + height - 1][column] = fill
    for row in range(matrix.y, matrix.y + height):
        canvas.background_colors[row][matrix.x] = fill
        canvas.background_colors[row][matrix.x + width - 1] = fill

    put(
        canvas,
        matrix.x + max(1, (width - len(matrix.columns)) // 2),
        matrix.y,
        matrix.columns,
        interior,
        fill,
    )
    row_label = "vocab" if matrix.key == "token_one_hot" else matrix.rows
    put(
        canvas,
        max(0, matrix.x - len(row_label) + 1),
        matrix.y + 1,
        row_label,
        exterior,
        matrix.panel_background,
    )
    put(
        canvas,
        matrix.x + (width - len(matrix.label)) // 2,
        matrix.y + height // 2,
        matrix.label,
        interior,
        fill,
        bold=True,
    )


def _subscript(value: int) -> str:
    return str(value).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _draw_merged_heads(
    canvas: Canvas,
    diagram: Diagram,
    matrix: Matrix,
    active: bool,
    result: bool,
) -> None:
    """Replace Hmerge's center label with first/ellipsis/last head labels."""
    _draw_matrix(canvas, diagram, matrix, active, result)
    fill = (
        RESULT_FILL
        if result
        else ACTIVE_INPUT_FILL
        if active
        else _inactive_fill(matrix.panel_background)
    )
    text = (
        RESULT_TEXT
        if result
        else ACTIVE_TEXT
        if active
        else INACTIVE_TEXT
    )
    exterior = ACTIVE_TEXT if active else INACTIVE_TEXT
    width = diagram.width_for(matrix.columns)
    height = diagram.height_for(matrix.rows)
    canvas.fill_rect(matrix.x + 1, matrix.y + 1, width - 2, height - 2, fill)

    config = diagram.config
    if config.attention_kind == "GQA":
        first = "H₀₀"
        last = (
            f"H{_subscript(config.kv_heads - 1)}"
            f"{_subscript(config.queries_per_kv - 1)}"
        )
        expanded = (
            f"{config.value_head_dim}×{config.queries_per_kv}×"
            f"{config.kv_heads}={config.merged_value_width}"
        )
    else:
        first = "H₀"
        last = f"H{_subscript(config.query_heads - 1)}"
        expanded = (
            f"{config.value_head_dim}×{config.query_heads}="
            f"{config.merged_value_width}"
        )

    for row, label in zip(
        range(matrix.y + 1, matrix.y + height - 1),
        (first, "⋮", last),
    ):
        put(
            canvas,
            matrix.x + (width - len(label)) // 2,
            row,
            label,
            text,
            fill,
            bold=True,
        )
    put(
        canvas,
        max(0, matrix.x - len(expanded) + 1),
        matrix.y + 1,
        expanded,
        exterior,
        matrix.panel_background,
    )


def _draw_attention_panels(canvas: Canvas, diagram: Diagram) -> None:
    panels = diagram.panels
    config = diagram.config
    shift = diagram.horizontal_shift
    level_one_x = LEVEL_ONE_X + shift
    level_two_x = LEVEL_TWO_X + shift

    if config.attention_kind == "MHA":
        _panel(
            canvas,
            level_one_x,
            panels.group_top,
            LEVEL_ONE_WIDTH,
            panels.group_bottom - panels.group_top,
            PEACH,
            f"MHA Head i ×{config.query_heads}",
            GROUP_BORDER,
        )
    elif config.attention_kind == "MQA":
        _panel(
            canvas,
            level_one_x,
            panels.group_top,
            LEVEL_ONE_WIDTH,
            panels.head_top - panels.group_top,
            PEACH,
            "Shared K/V",
            GROUP_BORDER,
        )
        _panel(
            canvas,
            level_one_x,
            panels.head_top,
            LEVEL_ONE_WIDTH,
            panels.group_bottom - panels.head_top,
            SKY,
            f"Query Head i ×{config.query_heads}",
            HEAD_BORDER,
        )
    else:
        _panel(
            canvas,
            level_one_x,
            panels.group_top,
            LEVEL_ONE_WIDTH,
            panels.group_bottom - panels.group_top,
            PEACH,
            f"GQA Group i ×{config.kv_heads}",
            GROUP_BORDER,
        )
        _panel(
            canvas,
            level_two_x,
            panels.head_top,
            LEVEL_TWO_WIDTH,
            panels.head_bottom - panels.head_top,
            SKY,
            f"Query Head j ×{config.queries_per_kv}",
            HEAD_BORDER,
        )


def _draw_end_labels(canvas: Canvas, diagram: Diagram) -> None:
    panels = diagram.panels
    config = diagram.config
    shift = diagram.horizontal_shift
    level_one_x = LEVEL_ONE_X + shift
    level_two_x = LEVEL_TWO_X + shift

    _end_label(
        canvas,
        x=0,
        bottom=panels.embedding_bottom - 1,
        width=diagram.block_width,
        name="Embedding",
        color=EMBEDDING_BORDER,
        background=EMBEDDING_BACKGROUND,
    )
    if config.attention_kind == "MHA":
        _end_label(
            canvas,
            x=level_one_x,
            bottom=panels.group_bottom - 1,
            width=LEVEL_ONE_WIDTH,
            name="MHA",
            color=GROUP_BORDER,
            background=PEACH,
        )
    elif config.attention_kind == "MQA":
        _end_label(
            canvas,
            x=level_one_x,
            bottom=panels.head_top - 1,
            width=LEVEL_ONE_WIDTH,
            name="Shared K/V",
            color=GROUP_BORDER,
            background=PEACH,
        )
        _end_label(
            canvas,
            x=level_one_x,
            bottom=panels.group_bottom - 1,
            width=LEVEL_ONE_WIDTH,
            name="Query",
            color=HEAD_BORDER,
            background=SKY,
        )
    else:
        _end_label(
            canvas,
            x=level_two_x,
            bottom=panels.head_bottom - 1,
            width=LEVEL_TWO_WIDTH,
            name="Head",
            color=HEAD_BORDER,
            background=SKY,
        )
        _end_label(
            canvas,
            x=level_one_x,
            bottom=panels.group_bottom - 1,
            width=LEVEL_ONE_WIDTH,
            name="GQA",
            color=GROUP_BORDER,
            background=PEACH,
        )

    _end_label(
        canvas,
        x=level_one_x,
        bottom=panels.mlp_bottom - 1,
        width=LEVEL_ONE_WIDTH,
        name="MLP",
        color=MLP_BORDER,
        background=MLP_BACKGROUND,
    )
    _end_label(
        canvas,
        x=0,
        bottom=panels.block_bottom - 1,
        width=diagram.block_width,
        name=config.family_name,
        color=BLOCK_BORDER,
        background=BLOCK_BACKGROUND,
    )
    _end_label(
        canvas,
        x=0,
        bottom=panels.canvas_height - 1,
        width=diagram.block_width,
        name="Final",
        color=FINAL_BORDER,
        background=PEACH,
    )


def _draw_card(
    canvas: Canvas,
    diagram: Diagram,
    step: int,
    *,
    annotation_x: int,
    card_y: int | None = None,
) -> None:
    """Draw the annotation beside the operation or at an explicit row."""
    title, operation, equation, source = card_lines(diagram, step)
    content = operation, equation, source
    width = maximum_card_width(diagram)
    y = (
        max(
            0,
            min(
                diagram.operations[step].row - 1,
                canvas.height - CARD_HEIGHT,
            ),
        )
        if card_y is None
        else card_y
    )
    canvas.fill_rect(annotation_x, y, width, CARD_HEIGHT, CARD_BACKGROUND)
    canvas.fancy_box(
        annotation_x, y, width, CARD_HEIGHT, "single", CARD_BORDER
    )
    put(
        canvas,
        annotation_x + 2,
        y,
        f" {title} ",
        CARD_BORDER,
        CARD_BACKGROUND,
        bold=True,
    )
    for offset, line in enumerate(content, start=1):
        label, value = line.split(": ", 1)
        x = annotation_x + 2
        put(
            canvas,
            x,
            y + offset,
            f"{label}:",
            CARD_LABEL,
            CARD_BACKGROUND,
            bold=True,
        )
        put(
            canvas,
            x + len(label) + 2,
            y + offset,
            value,
            CARD_TEXT,
            CARD_BACKGROUND,
        )


def render_diagram(
    diagram: Diagram,
    step: int,
    *,
    annotation_position: str = "right",
) -> Canvas:
    """Render one state with the card to the right or below the model."""
    if annotation_position not in {"right", "below"}:
        raise ValueError("annotation_position must be 'right' or 'below'")
    card_width = maximum_card_width(diagram)
    if annotation_position == "right":
        canvas_width = diagram.block_width + 1 + card_width + 1
        canvas_height = diagram.panels.canvas_height
    else:
        canvas_width = max(diagram.block_width, card_width)
        canvas_height = diagram.panels.canvas_height + 1 + CARD_HEIGHT
    canvas = Canvas(canvas_width, canvas_height)
    panels = diagram.panels
    config = diagram.config

    _panel(
        canvas,
        0,
        0,
        diagram.block_width,
        panels.embedding_bottom,
        EMBEDDING_BACKGROUND,
        "Embedding",
        EMBEDDING_BORDER,
    )
    _panel(
        canvas,
        0,
        panels.block_top,
        diagram.block_width,
        panels.block_bottom - panels.block_top,
        BLOCK_BACKGROUND,
        f"{config.family_name} Block b ×{config.block_count}",
        BLOCK_BORDER,
    )
    _draw_attention_panels(canvas, diagram)
    _panel(
        canvas,
        LEVEL_ONE_X + diagram.horizontal_shift,
        panels.mlp_top,
        LEVEL_ONE_WIDTH,
        panels.mlp_bottom - panels.mlp_top,
        MLP_BACKGROUND,
        "MLP",
        MLP_BORDER,
    )
    _panel(
        canvas,
        0,
        panels.final_top,
        diagram.block_width,
        panels.canvas_height - panels.final_top,
        PEACH,
        "Final",
        FINAL_BORDER,
    )

    operation = diagram.operations[step]
    for matrix in diagram.matrices:
        active = matrix.key in operation.matrices
        result = matrix.key == operation.result
        if matrix.key == "h_merge":
            _draw_merged_heads(canvas, diagram, matrix, active, result)
        else:
            _draw_matrix(canvas, diagram, matrix, active, result)

    # End labels are drawn after matrices so they always remain legible on the
    # calculated border rows.
    _draw_end_labels(canvas, diagram)
    if annotation_position == "right":
        _draw_card(
            canvas,
            diagram,
            step,
            annotation_x=diagram.block_width + 1,
        )
    else:
        _draw_card(
            canvas,
            diagram,
            step,
            annotation_x=0,
            card_y=diagram.panels.canvas_height + 1,
        )
    return canvas
