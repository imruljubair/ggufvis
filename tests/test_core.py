"""Focused tests for the standalone final-v1 implementation."""

from __future__ import annotations

import ast
import http.server
import os
import re
import struct
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ggufvis.cli import _viewport
from ggufvis.annotations import card_lines
from ggufvis.diagram import build_diagram
from ggufvis.gguf import GGUFModel, TensorInfo, read_gguf_url
from ggufvis.model import config_from_gguf
from ggufvis.renderer import render_diagram


def _tensor(name: str, *shape: int) -> TensorInfo:
    return TensorInfo(name, shape, 0, 0)


def _qwen3_model(*, tied: bool = False) -> GGUFModel:
    hidden, head_dim, ffn, vocab = 4096, 128, 12288, 151936
    metadata = {
        "general.architecture": "qwen3",
        "general.name": "Synthetic Qwen3 8B",
        "qwen3.block_count": 36,
        "qwen3.context_length": 40960,
        "qwen3.embedding_length": hidden,
        "qwen3.feed_forward_length": ffn,
        "qwen3.attention.head_count": 32,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": head_dim,
        "qwen3.attention.value_length": head_dim,
        "qwen3.vocab_size": vocab,
    }
    tensors = [
        _tensor("token_embd.weight", hidden, vocab),
        _tensor("blk.0.attn_q.weight", hidden, 32 * head_dim),
        _tensor("blk.0.attn_k.weight", hidden, 8 * head_dim),
        _tensor("blk.0.attn_v.weight", hidden, 8 * head_dim),
        _tensor("blk.0.attn_output.weight", hidden, 32 * head_dim),
        _tensor("blk.0.attn_q_norm.weight", head_dim),
        _tensor("blk.0.attn_k_norm.weight", head_dim),
        _tensor("blk.0.ffn_gate.weight", hidden, ffn),
        _tensor("blk.0.ffn_up.weight", hidden, ffn),
        _tensor("blk.0.ffn_down.weight", ffn, hidden),
        _tensor("blk.35.attn_norm.weight", hidden),
    ]
    if not tied:
        tensors.append(_tensor("output.weight", hidden, vocab))
    return GGUFModel(
        Path("synthetic-qwen3.gguf"), 3, metadata, tuple(tensors)
    )


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _remote_fixture() -> bytes:
    """Create a tiny valid header followed by a large fake tensor payload."""
    header = bytearray(b"GGUF")
    header.extend(struct.pack("<IQQ", 3, 1, 2))
    for key, value in (
        ("general.architecture", "llama"),
        ("general.name", "Remote Test"),
    ):
        header.extend(_gguf_string(key))
        header.extend(struct.pack("<I", 8))  # GGUF string value type
        header.extend(_gguf_string(value))
    header.extend(_gguf_string("token_embd.weight"))
    header.extend(struct.pack("<IQQIQ", 2, 4, 8, 0, 0))
    return bytes(header) + b"\0" * (2 * 1024 * 1024)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    data = _remote_fixture()
    requested_ranges: list[tuple[int, int]] = []

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if match is None:
            self.send_error(400, "Range required")
            return
        start, requested_end = map(int, match.groups())
        if start >= len(self.data):
            self.send_response(416)
            self.end_headers()
            return
        end = min(requested_end, len(self.data) - 1)
        self.requested_ranges.append((start, end))
        payload = self.data[start : end + 1]
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header(
            "Content-Range", f"bytes {start}-{end}/{len(self.data)}"
        )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        pass


class FinalV1Tests(unittest.TestCase):
    def test_qwen_dimensions_norms_and_annotations_are_integrated(self) -> None:
        config = config_from_gguf(_qwen3_model())
        diagram = build_diagram(config)
        result_keys = [operation.result for operation in diagram.operations]
        step = result_keys.index("k_norm")
        lines = card_lines(diagram, step)
        rendered = render_diagram(diagram, step).render(use_color=False)

        self.assertEqual(config.attention_kind, "GQA")
        self.assertEqual(len(diagram.operations), 23)
        self.assertEqual(diagram.height_for("seq"), 7)
        self.assertEqual(diagram.height_for("151936"), 7)
        self.assertEqual(lines[0], "[04/23] ATTENTION")
        self.assertIn("K′[128×seq]", lines[2])
        self.assertEqual(
            lines[3],
            "Tensor source: blk.b.attn_k_norm.weight",
        )
        self.assertIn(" Qwen3 Block b ×36 ", rendered)
        self.assertIn(" Head end ┘", rendered)
        self.assertIn(" GQA end ┘", rendered)

    def test_tied_output_uses_embedding_source(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model(tied=True)))
        lines = card_lines(diagram, len(diagram.operations) - 1)

        self.assertEqual(diagram.matrix("wlm").label, "Eᵀ")
        self.assertEqual(
            lines[3],
            "Tensor source: token_embd.weight (transposed, tied)",
        )

    def test_remote_reader_uses_ranges_without_saving_the_model(self) -> None:
        _RangeHandler.requested_ranges = []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = (
            f"http://127.0.0.1:{server.server_address[1]}/remote-test.gguf"
        )

        model = read_gguf_url(url)

        self.assertEqual(model.metadata["general.architecture"], "llama")
        self.assertEqual(model.tensor("token_embd.weight").shape, (4, 8))
        self.assertEqual(model.remote_file_size, len(_RangeHandler.data))
        self.assertLess(model.remote_bytes_transferred, len(_RangeHandler.data))
        self.assertTrue(_RangeHandler.requested_ranges)
        self.assertFalse(model.path.exists())

    def test_annotation_can_stack_below_for_narrow_terminals(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        beside = render_diagram(diagram, 0, annotation_position="right")
        below = render_diagram(diagram, 0, annotation_position="below")
        below_text = below.render(use_color=False)

        self.assertLess(below.width, beside.width)
        self.assertGreater(below.height, beside.height)
        self.assertIn("[01/23] EMBEDDING", below_text)
        self.assertIn("Tensor source: token_embd.weight", below_text)

    def test_narrow_viewport_keeps_annotation_visible(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        with patch(
            "ggufvis.cli.shutil.get_terminal_size",
            return_value=os.terminal_size((81, 30)),
        ):
            frame, _ = _viewport(diagram, 0, False, None)

        self.assertNotIn("Widen the window", frame)
        self.assertIn("[01/23] EMBEDDING", frame)
        self.assertIn("Tensor source: token_embd.weight", frame)
        self.assertIn(
            "Tip: widen the terminal to place annotations beside the diagram.",
            frame,
        )

    def test_mha_and_mqa_have_clean_non_nested_labels(self) -> None:
        base = config_from_gguf(_qwen3_model())
        mha = replace(
            base,
            qk_norm=False,
            query_heads=16,
            kv_heads=16,
            key_head_dim=256,
            value_head_dim=256,
        )
        mqa = replace(
            base,
            qk_norm=False,
            query_heads=16,
            kv_heads=1,
            key_head_dim=256,
            value_head_dim=256,
        )
        mha_text = render_diagram(build_diagram(mha), 0).render(False)
        mqa_text = render_diagram(build_diagram(mqa), 0).render(False)

        self.assertIn(" MHA end ┘", mha_text)
        self.assertNotIn(" Head end ┘", mha_text)
        self.assertIn(" Shared K/V end ┘", mqa_text)
        self.assertIn(" Query end ┘", mqa_text)

    def test_final_folder_has_no_historical_visualizer_imports(self) -> None:
        folder = Path(__file__).parents[1] / "ggufvis"
        historical = {
            f"ggufvis{version}" for version in range(2, 9)
        } | {"tensor_diagram_interactive_full_block_external_annotations"}
        imported: set[str] = set()
        for path in folder.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)

        self.assertTrue(historical.isdisjoint(imported))


if __name__ == "__main__":
    unittest.main()
