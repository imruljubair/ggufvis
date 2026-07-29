"""Command-line and interactive navigation for final v1."""

from __future__ import annotations

import argparse
import shutil
import sys
import termios
import tty
from pathlib import Path

from .diagram import Diagram, build_diagram
from .gguf import GGUFError, read_gguf, read_gguf_url, resolve_ollama_model
from .model import config_from_gguf
from .renderer import render_diagram
from .terminal import read_key


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Render a standalone metadata-driven matrix diagram for dense "
            "Llama, original Gemma, Qwen2/Qwen2.5, Qwen3, or DeepSeek-R1 "
            "Distill GGUF models."
        )
    )
    result.add_argument(
        "gguf",
        help=(
            "local GGUF path, Ollama model name with --ollama, or direct "
            "HTTP(S) GGUF link with --url"
        ),
    )
    source_mode = result.add_mutually_exclusive_group()
    source_mode.add_argument(
        "--ollama",
        action="store_true",
        help="resolve the positional argument as an Ollama model name",
    )
    source_mode.add_argument(
        "--url",
        action="store_true",
        help=(
            "parse a remote GGUF through in-memory HTTP range requests "
            "without saving the model"
        ),
    )
    result.add_argument(
        "--ollama-models-dir",
        metavar="PATH",
        help=(
            "Ollama models directory; defaults to OLLAMA_MODELS or "
            "~/.ollama/models"
        ),
    )
    result.add_argument("--no-color", action="store_true")
    result.add_argument(
        "--static",
        action="store_true",
        help="print one operation without entering interactive mode",
    )
    result.add_argument(
        "--step",
        type=int,
        default=1,
        help="1-based operation used by --static (default: 1)",
    )
    return result


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _viewport(
    diagram: Diagram,
    step: int,
    use_color: bool,
    requested_start: int | None,
) -> tuple[str, int]:
    terminal = shutil.get_terminal_size(
        fallback=(120, diagram.panels.canvas_height + 1)
    )
    canvas = render_diagram(diagram, step, annotation_position="right")
    stacked = False
    if terminal.columns < canvas.width:
        # The annotation is wider than the model but still fits comfortably
        # below it in ordinary 80-column terminals.
        canvas = render_diagram(diagram, step, annotation_position="below")
        stacked = True
    if terminal.columns < canvas.width:
        return (
            f"Terminal is {terminal.columns} columns wide; this diagram needs "
            f"at least {canvas.width}. Widen the window, then press an arrow key.",
            0,
        )

    lines = canvas.render(use_color=use_color).splitlines()
    if stacked:
        # Keep the below-diagram card visible while the matrix viewport moves.
        # The final six rows are one spacer plus the five-row card.
        model_lines = lines[: diagram.panels.canvas_height]
        card_lines = lines[diagram.panels.canvas_height :]
        viewport_height = min(
            len(model_lines),
            # Reserve one row for status and one for the narrow-layout tip.
            max(4, terminal.lines - len(card_lines) - 2),
        )
        last_start = max(0, len(model_lines) - viewport_height)
    else:
        model_lines = lines
        card_lines = []
        viewport_height = min(len(lines), max(4, terminal.lines - 1))
        last_start = max(0, len(lines) - viewport_height)
    if requested_start is None:
        centered = diagram.operations[step].row - viewport_height // 2
        start = max(0, min(centered, last_start))
    else:
        start = max(0, min(requested_start, last_start))
    end = start + viewport_height
    visible_lines = model_lines[start:end]
    if card_lines:
        visible_lines.extend(card_lines)
    visible = "\n".join(visible_lines)
    status = (
        f"Rows {start + 1}-{end}/{len(model_lines)}  "
        "Up/Down: step  PgUp/PgDn: view  q: quit"
    )
    if stacked:
        tip = (
            "Tip: widen the terminal to place annotations beside the diagram."
        )
        return f"{visible}\n{tip}\n{status}", start
    return f"{visible}\n{status}", start


def _run_interactive(diagram: Diagram, use_color: bool) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(render_diagram(diagram, 0).render(use_color=use_color))
        print("Interactive mode requires a terminal. Use --static for one frame.")
        return

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    step = 0
    viewport_start: int | None = None
    try:
        tty.setraw(descriptor)
        sys.stdout.write("\033[?1049h\033[?25l")
        while True:
            frame, actual_start = _viewport(
                diagram, step, use_color, viewport_start
            )
            sys.stdout.write(
                f"\033[2J\033[H{frame.replace(chr(10), chr(13) + chr(10))}"
            )
            sys.stdout.flush()
            key = read_key()
            if key == "down":
                step = min(step + 1, len(diagram.operations) - 1)
                viewport_start = None
            elif key == "up":
                step = max(step - 1, 0)
                viewport_start = None
            elif key == "page_up":
                viewport_start = actual_start - 10
            elif key == "page_down":
                viewport_start = actual_start + 10
            elif key.lower() == "q" or key == "\x03":
                break
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.ollama_models_dir and not args.ollama:
            raise GGUFError("--ollama-models-dir requires --ollama")
        if args.url:
            model = read_gguf_url(args.gguf)
        else:
            path = (
                resolve_ollama_model(args.gguf, args.ollama_models_dir)
                if args.ollama
                else Path(args.gguf).expanduser()
            )
            model = read_gguf(path)
        if model.source_url:
            print(
                "remote: parsed metadata and "
                f"{len(model.tensors)} tensor descriptors; transferred "
                f"{_human_bytes(model.remote_bytes_transferred)} of "
                f"{_human_bytes(model.remote_file_size)}; no local model "
                "file created",
                file=sys.stderr,
            )
        config = config_from_gguf(model)
        diagram = build_diagram(config)
        if not 1 <= args.step <= len(diagram.operations):
            raise GGUFError(
                f"--step must be between 1 and {len(diagram.operations)}"
            )
        for warning in config.warnings:
            print(f"warning: {warning}", file=sys.stderr)

        use_color = not args.no_color
        if args.static:
            print(
                render_diagram(diagram, args.step - 1).render(
                    use_color=use_color
                )
            )
        else:
            _run_interactive(diagram, use_color)
    except GGUFError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
