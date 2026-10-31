"""Self-contained HTML export for the terminal diagram."""

from __future__ import annotations

import html
import json

from .diagram import Diagram
from .renderer import render_diagram
from .terminal import Canvas, Color


# Browser equivalents of the conventional 16-color terminal palette. Most of
# the diagram uses explicit RGB tuples; this table covers its named accents.
ANSI_RGB = {
    "black": (0, 0, 0),
    "red": (205, 49, 49),
    "green": (13, 188, 121),
    "yellow": (229, 229, 16),
    "blue": (36, 114, 200),
    "magenta": (188, 63, 188),
    "cyan": (17, 168, 205),
    "white": (229, 229, 229),
    "bright_black": (102, 102, 102),
    "bright_red": (241, 76, 76),
    "bright_green": (35, 209, 139),
    "bright_yellow": (245, 245, 67),
    "bright_blue": (59, 142, 234),
    "bright_magenta": (214, 112, 214),
    "bright_cyan": (41, 184, 219),
    "bright_white": (255, 255, 255),
}


def _css_color(color: Color) -> str | None:
    if color is None:
        return None
    rgb = color if isinstance(color, tuple) else ANSI_RGB[color]
    return f"rgb({rgb[0]} {rgb[1]} {rgb[2]})"


def canvas_fragment(canvas: Canvas) -> str:
    """Return one Canvas as compact, styled HTML with one span per style run."""
    rows: list[str] = []
    for pixels, colors, backgrounds, bolds in zip(
        canvas.pixels,
        canvas.colors,
        canvas.background_colors,
        canvas.bold,
    ):
        runs: list[str] = []
        start = 0
        active = colors[0], backgrounds[0], bolds[0]
        for index in range(1, canvas.width + 1):
            style = (
                (colors[index], backgrounds[index], bolds[index])
                if index < canvas.width
                else None
            )
            if style == active:
                continue
            foreground, background, is_bold = active
            declarations: list[str] = []
            if (value := _css_color(foreground)) is not None:
                declarations.append(f"color:{value}")
            if (value := _css_color(background)) is not None:
                declarations.append(f"background:{value}")
            if is_bold:
                declarations.append("font-weight:700")
            content = html.escape("".join(pixels[start:index]))
            if declarations:
                runs.append(
                    f'<span style="{";".join(declarations)}">{content}</span>'
                )
            else:
                runs.append(content)
            start = index
            active = style
        rows.append("".join(runs))
    return "\n".join(rows)


def render_html(
    diagram: Diagram,
    source_header: str,
    *,
    initial_step: int = 0,
) -> str:
    """Render every operation into one dependency-free interactive document."""
    if diagram.config.architecture != "llama":
        raise ValueError("HTML export currently supports Llama models only")
    if not 0 <= initial_step < len(diagram.operations):
        raise ValueError("initial HTML step is out of range")

    frames = [
        canvas_fragment(render_diagram(diagram, step))
        for step in range(len(diagram.operations))
    ]
    # Escaping '<' prevents model-provided text from ending the script element.
    frame_json = json.dumps(frames, ensure_ascii=False).replace("<", "\\u003c")
    title = html.escape(f"{diagram.config.name} · ggufvis")
    source = html.escape(source_header)
    family = html.escape(diagram.config.family_name)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: #0c0d0f;
      color: #e5e5e5;
      font-family: ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas,
        "Liberation Mono", "DejaVu Sans Mono", monospace;
    }
    .shell { min-height: 100vh; display: flex; flex-direction: column; }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 58px;
      padding: 10px 16px;
      background: rgb(17 18 21 / 94%);
      border-bottom: 1px solid #303238;
      backdrop-filter: blur(10px);
    }
    .identity { min-width: 210px; margin-right: auto; }
    .identity strong { display: block; color: #fff; font-size: 14px; }
    .identity span { color: #9699a1; font-size: 12px; }
    button {
      min-width: 42px;
      height: 34px;
      padding: 0 12px;
      border: 1px solid #494c54;
      border-radius: 6px;
      background: #22242a;
      color: #f0f0f0;
      font: inherit;
      cursor: pointer;
    }
    button:hover:not(:disabled), button:focus-visible {
      border-color: rgb(235 70 205);
      outline: none;
    }
    button:disabled { cursor: default; opacity: .35; }
    #position { min-width: 70px; text-align: center; font-size: 13px; }
    .viewport { flex: 1; overflow: auto; padding: 18px; }
    .terminal {
      display: inline-block;
      min-width: max-content;
      margin: 0;
      padding: 0;
      color: #e5e5e5;
      background: transparent;
      font: 14px/1 ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas,
        "Liberation Mono", "DejaVu Sans Mono", monospace;
      font-variant-ligatures: none;
      letter-spacing: 0;
      tab-size: 1;
      white-space: pre;
    }
    .hint { color: #737780; font-size: 12px; white-space: nowrap; }
    @media (max-width: 720px) {
      .toolbar { flex-wrap: wrap; gap: 8px; }
      .identity { flex: 1 0 100%; }
      .hint { margin-left: auto; }
      .viewport { padding: 12px; }
      .terminal { font-size: 12px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="toolbar" aria-label="Diagram navigation">
      <div class="identity">
        <strong>__FAMILY__ visualization</strong>
        <span>__SOURCE__</span>
      </div>
      <button id="previous" type="button" aria-label="Previous operation">↑</button>
      <span id="position" aria-live="polite"></span>
      <button id="next" type="button" aria-label="Next operation">↓</button>
      <span class="hint">Use ↑ and ↓ to step</span>
    </nav>
    <section class="viewport" aria-label="Scrollable model diagram">
      <pre id="terminal" class="terminal" aria-label="Llama matrix visualization"></pre>
    </section>
  </main>
  <script>
    const frames = __FRAMES__;
    let step = __INITIAL_STEP__;
    const terminal = document.querySelector("#terminal");
    const position = document.querySelector("#position");
    const previous = document.querySelector("#previous");
    const next = document.querySelector("#next");

    function show(requested) {
      step = Math.max(0, Math.min(frames.length - 1, requested));
      terminal.innerHTML = frames[step];
      position.textContent = `${step + 1} / ${frames.length}`;
      previous.disabled = step === 0;
      next.disabled = step === frames.length - 1;
    }

    previous.addEventListener("click", () => show(step - 1));
    next.addEventListener("click", () => show(step + 1));
    document.addEventListener("keydown", (event) => {
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault();
        show(step - 1);
      } else if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        event.preventDefault();
        show(step + 1);
      } else if (event.key === "Home") {
        event.preventDefault();
        show(0);
      } else if (event.key === "End") {
        event.preventDefault();
        show(frames.length - 1);
      }
    });
    show(step);
  </script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", title)
        .replace("__FAMILY__", family)
        .replace("__SOURCE__", source)
        .replace("__FRAMES__", frame_json)
        .replace("__INITIAL_STEP__", str(initial_step))
    )
