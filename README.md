# ggufvis

A metadata-driven terminal matrix visualizer for GGUF language models.

The visualizer reads GGUF metadata and tensor descriptors only. It never reads
or displays tensor weight values, so opening a large model does not load its
weights into memory.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://github.com/imruljubair/ggufvis/actions/workflows/tests.yml/badge.svg)

## Features

- Matrix-oriented transformer visualization with interactive stepping
- Dimensions, head counts, and block counts derived from GGUF
- MHA, GQA, and MQA attention layouts
- Llama, original Gemma, Qwen2/Qwen2.5, and Qwen3 support
- Qwen Q/K normalization and projection-bias detection
- Tied and untied language-model heads
- Shape-aware annotation cards with canonical GGUF tensor sources
- Direct access to installed Ollama models through `--ollama`
- No runtime dependencies outside the Python standard library

## Run

Clone the repository and run directly:

```bash
git clone https://github.com/imruljubair/ggufvis.git
cd ggufvis

python3 ggufvis.py /path/to/model.gguf
python3 ggufvis.py --ollama qwen3
```

Or install an editable command:

```bash
python3 -m pip install -e .
ggufvis --ollama qwen3
```

Render one frame without entering interactive mode:

```bash
python3 ggufvis.py --ollama qwen3 --static --step 4
python3 ggufvis.py --ollama qwen3 --static --step 4 --no-color
```

Interactive keys:

- `Down`: next operation
- `Up`: previous operation
- `Page Down` / `Page Up`: move the viewport
- `q`: quit

On a wide terminal, the annotation card appears beside the matrix view. On a
narrow terminal, interactive mode automatically places the card below the
diagram instead.

Ollama resolution is enabled only by `--ollama`. A normal positional argument
is always interpreted as a filesystem path.

## Supported architectures

- Dense Llama
- Original Gemma
- Dense Qwen2 and Qwen2.5 (`general.architecture = qwen2`)
- Dense Qwen3

MoE architectures and later Gemma architectures are rejected explicitly
instead of being rendered as if they were structurally identical.

## Source layout

```text
ggufvis.py              repository-local launcher
ggufvis/
  __main__.py           `python -m ggufvis` entry point
  cli.py                arguments, viewport, keyboard navigation
  gguf.py               GGUF reader and Ollama model resolution
  model.py              architecture and dimension normalization
  diagram.py            matrix and panel geometry
  annotations.py        equations and GGUF tensor sources
  renderer.py           terminal visualization
  terminal.py           local ANSI/Unicode canvas
tests/
  test_core.py          standalone unit tests
```

The dependency direction is straightforward:

```text
CLI
 ├─ GGUF/Ollama reader
 ├─ model normalization
 ├─ diagram layout
 └─ renderer
     ├─ annotations
     └─ terminal canvas
```

There are no imports from historical prototype versions.

## Data flow

1. `gguf.py` reads metadata and tensor descriptors.
2. `model.py` normalizes architecture-specific metadata into `ModelConfig`.
3. `diagram.py` pre-calculates all dimensions, matrix positions, and borders.
4. `annotations.py` creates the four-part card for each operation.
5. `renderer.py` draws one highlighted operation.
6. `cli.py` handles static output or interactive navigation.

## Dimension-to-height encoding

Matrix height represents dimension ordering, not literal proportional size.
For the currently installed Qwen3 8B model:

```text
1      → 3
128    → 4
4096   → 5
12288  → 6
seq    → 7
151936 → 7
```

Unique structural dimensions are sorted dynamically. Vocabulary height always
reuses the calculated sequence height.

## Annotation card

Every operation uses the same four fields:

```text
┌─ [04/23] ATTENTION ────────────────────────────────────┐
│ Operation: K per-head RMSNorm                          │
│ Equation: K′[128×seq] = RMSNorm(K[128×seq]; γk[128×1]) │
│ Tensor source: blk.b.attn_k_norm.weight                │
└────────────────────────────────────────────────────────┘
```

`blk.b` means the corresponding tensor in each repeated transformer block.
Parameter-free operations say `none (parameter-free)`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests include an explicit check that production source does not import a
historical prototype module.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local development and pull
request workflow.
