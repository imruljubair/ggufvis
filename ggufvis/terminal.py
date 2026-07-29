"""Minimal terminal drawing primitives used by the visualizer.

This module intentionally contains only the canvas features needed by the
matrix renderer.  It has no dependency on the earlier visualization scripts.
"""

from __future__ import annotations

import sys


Color = str | tuple[int, int, int] | None

COLORS = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "bright_black": 90,
    "bright_red": 91,
    "bright_green": 92,
    "bright_yellow": 93,
    "bright_blue": 94,
    "bright_magenta": 95,
    "bright_cyan": 96,
    "bright_white": 97,
}

BOX_STYLES = {
    "single": ("┌", "┐", "└", "┘", "─", "│"),
    "double": ("╔", "╗", "╚", "╝", "═", "║"),
    "heavy": ("┏", "┓", "┗", "┛", "━", "┃"),
}


class Canvas:
    """A character grid with independent foreground/background styles."""

    def __init__(self, width: int, height: int) -> None:
        if width < 1 or height < 1:
            raise ValueError("canvas dimensions must be positive")
        self.width = width
        self.height = height
        self.pixels = [[" " for _ in range(width)] for _ in range(height)]
        self.colors: list[list[Color]] = [
            [None for _ in range(width)] for _ in range(height)
        ]
        self.background_colors: list[list[Color]] = [
            [None for _ in range(width)] for _ in range(height)
        ]
        self.bold = [[False for _ in range(width)] for _ in range(height)]

    def point(
        self,
        x: int,
        y: int,
        char: str = " ",
        color: Color = None,
        background: Color = None,
        bold: bool = False,
    ) -> None:
        """Set one cell; out-of-range points are harmless."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = char[0]
            self.colors[y][x] = color
            self.background_colors[y][x] = background
            self.bold[y][x] = bold

    def fill_rect(
        self, x: int, y: int, width: int, height: int, background: Color
    ) -> None:
        for row in range(y, y + height):
            for column in range(x, x + width):
                self.point(column, row, " ", None, background)

    def line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        char: str,
        color: Color,
    ) -> None:
        """Draw the horizontal/vertical lines used by boxes."""
        if y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.point(x, y1, char, color)
            return
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.point(x1, y, char, color)
            return
        raise ValueError("terminal canvas lines must be horizontal or vertical")

    def fancy_box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: str,
        color: Color,
    ) -> None:
        """Draw a Unicode single, double, or heavy box."""
        tl, tr, bl, br, horizontal, vertical = BOX_STYLES[style]
        self.line(x + 1, y, x + width - 2, y, horizontal, color)
        self.line(
            x + 1,
            y + height - 1,
            x + width - 2,
            y + height - 1,
            horizontal,
            color,
        )
        self.line(x, y + 1, x, y + height - 2, vertical, color)
        self.line(
            x + width - 1,
            y + 1,
            x + width - 1,
            y + height - 2,
            vertical,
            color,
        )
        self.point(x, y, tl, color)
        self.point(x + width - 1, y, tr, color)
        self.point(x, y + height - 1, bl, color)
        self.point(x + width - 1, y + height - 1, br, color)

    def render(self, use_color: bool = True) -> str:
        """Return the grid as plain text or true-color ANSI text."""
        lines: list[str] = []
        for pixels, colors, backgrounds, bolds in zip(
            self.pixels, self.colors, self.background_colors, self.bold
        ):
            last = next(
                (
                    index
                    for index in range(self.width - 1, -1, -1)
                    if pixels[index] != " " or backgrounds[index] is not None
                ),
                -1,
            )
            if last < 0:
                lines.append("")
                continue
            if not use_color:
                lines.append("".join(pixels[: last + 1]))
                continue

            output: list[str] = []
            active: tuple[Color, Color, bool] = (None, None, False)
            for char, foreground, background, is_bold in zip(
                pixels[: last + 1],
                colors[: last + 1],
                backgrounds[: last + 1],
                bolds[: last + 1],
            ):
                style = foreground, background, is_bold
                if style != active:
                    codes = ["0"]
                    if is_bold:
                        codes.append("1")
                    if foreground is not None:
                        if isinstance(foreground, tuple):
                            codes.append(
                                f"38;2;{foreground[0]};{foreground[1]};"
                                f"{foreground[2]}"
                            )
                        else:
                            codes.append(str(COLORS.get(foreground, 37)))
                    if background is not None:
                        if isinstance(background, tuple):
                            codes.append(
                                f"48;2;{background[0]};{background[1]};"
                                f"{background[2]}"
                            )
                        else:
                            codes.append(str(COLORS.get(background, 30) + 10))
                    output.append(f"\033[{';'.join(codes)}m")
                    active = style
                output.append(char)
            if active != (None, None, False):
                output.append("\033[0m")
            lines.append("".join(output))
        return "\n".join(lines)


def put(
    canvas: Canvas,
    x: int,
    y: int,
    text: str,
    color: Color,
    background: Color,
    *,
    bold: bool = False,
) -> None:
    """Write styled text without requiring a second text abstraction."""
    for offset, char in enumerate(text):
        canvas.point(x + offset, y, char, color, background, bold)


def read_key() -> str:
    """Read one normal, arrow, or page-navigation key."""
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    second = sys.stdin.read(1)
    if second != "[":
        return "escape"
    third = sys.stdin.read(1)
    if third in {"5", "6"} and sys.stdin.read(1) == "~":
        return {"5": "page_up", "6": "page_down"}[third]
    return {"A": "up", "B": "down"}.get(third, "")
