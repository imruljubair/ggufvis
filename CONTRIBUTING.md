# Contributing

Contributions should preserve the central rule of this project: visualization
uses GGUF metadata and tensor descriptors, never tensor weight values.

## Development setup

```bash
git clone https://github.com/imruljubair/ggufvis.git
cd ggufvis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The project has no runtime dependencies outside the Python standard library.

## Tests

```bash
python -m unittest discover -s tests -v
```

Before opening a pull request, also test one local GGUF when possible:

```bash
python ggufvis.py /path/to/model.gguf --static --no-color
python ggufvis.py --ollama qwen3 --static --no-color
```

Do not add GGUF model files to the repository. They are excluded by
`.gitignore`.

Remote-reader changes must pass the synthetic local HTTP range-server test.
Tests and CI must not depend on external model hosting being available.

## Architecture additions

New model families should:

1. Be selected from `general.architecture`, not from the display name.
2. Derive dimensions from metadata and tensor descriptors.
3. Validate tensor shapes without reading tensor data.
4. Add architecture-specific matrices only when their tensors exist.
5. Include synthetic tests that require no downloaded model.

## Pull requests

Keep changes focused and explain:

- which GGUF architecture identifiers are affected;
- which metadata keys and tensor names are used;
- how matrix geometry changes;
- which local model tags were tested.
