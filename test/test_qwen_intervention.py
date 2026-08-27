#!/usr/bin/env python3
"""Offline tests for Qwen residual-writer direction extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qwen_intervention as qi  # noqa: E402
import qwen_quantize  # noqa: E402


def write_corpus(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record_id, content in rows:
            stream.write(json.dumps({
                "id": record_id,
                "messages": [{"role": "user", "content": content}],
            }, sort_keys=True) + "\n")


class FakeHandle:
    def __init__(self, point: "FakeHookPoint", hook):
        self.point = point
        self.hook = hook

    def remove(self) -> None:
        self.point.hooks.remove(self.hook)


class FakeHookPoint:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self, hook)

    def emit(self, writer_output) -> None:
        for hook in list(self.hooks):
            hook(self, (), writer_output)


class FakeQwen:
    def __init__(self, layer_count: int, qsa_layers: frozenset[int]):
        self.layers = [
            SimpleNamespace(
                linear_attn=SimpleNamespace(out_proj=FakeHookPoint()),
                self_attn=SimpleNamespace(o_proj=FakeHookPoint()),
            )
            for _ in range(layer_count)
        ]
        self.qsa_layers = qsa_layers
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=self.layers)
        )
        self.saw_output_hidden_states = False

    def __call__(self, *, layer_outputs, use_cache, return_dict, **kwargs):
        self.saw_output_hidden_states = "output_hidden_states" in kwargs
        if use_cache is not False or return_dict is not True:
            raise AssertionError("extractor forward contract changed")
        for index, (layer, output) in enumerate(
                zip(self.layers, layer_outputs, strict=True)):
            writer = (layer.self_attn.o_proj if index in self.qsa_layers
                      else layer.linear_attn.out_proj)
            writer.emit(output)
        return {"ignored": True}


class QwenInterventionTests(unittest.TestCase):
    def test_writer_output_directions_require_every_layer(self) -> None:
        spec = qi.ArchitectureSpec(
            layer_count=2, hidden_size=3, writer_input_size=6,
            qsa_layers=frozenset(),
        )
        good = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        bad = [[1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
        directions = qi.build_directions(
            good, bad, orthogonalize_control_mean=True, spec=spec,
        )
        self.assertEqual(len(directions), 2)
        self.assertTrue(all(row is not None for row in directions))

        degenerate_layer_zero = [list(row) for row in bad]
        degenerate_layer_zero[0] = list(good[0])
        with self.assertRaisesRegex(qi.InterventionError, "layer 0 bad-good direction"):
            qi.build_directions(
                good, degenerate_layer_zero,
                orthogonalize_control_mean=True, spec=spec,
            )

    def test_mock_hooks_capture_hybrid_residual_writer_outputs(self) -> None:
        spec = qi.ArchitectureSpec(
            layer_count=4, hidden_size=3, writer_input_size=6,
            qsa_layers=frozenset({3}),
        )
        model = FakeQwen(spec.layer_count, spec.qsa_layers)
        # Each layer contains batch x position x hidden.  The first position is
        # deliberately huge: selecting ordinary/initial state instead of the
        # final writer-output frontier would make every assertion fail.
        layer_outputs = []
        for layer in range(spec.layer_count):
            layer_outputs.append([
                [[1000.0, 1000.0, 1000.0], [layer + 1.0, 2.0, 3.0]],
                [[2000.0, 2000.0, 2000.0], [layer + 3.0, 4.0, 5.0]],
            ])
        means, count = qi.accumulate_activation_means(
            model, [({"layer_outputs": layer_outputs}, 2)], spec=spec,
            tensor_ops=qi.PythonTensorOps(), winsorization_quantile=1.0,
        )
        self.assertEqual(count, 2)
        for layer in range(spec.layer_count):
            self.assertEqual(means[layer], [layer + 2.0, 3.0, 4.0])
            self.assertEqual(model.layers[layer].linear_attn.out_proj.hooks, [])
            self.assertEqual(model.layers[layer].self_attn.o_proj.hooks, [])
        self.assertFalse(model.saw_output_hidden_states)

    def test_raw_dump_is_streamed_strictly_and_averaged(self) -> None:
        spec = qi.ArchitectureSpec(
            layer_count=2, hidden_size=3, writer_input_size=6,
            qsa_layers=frozenset(),
        )
        records = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "activations.f32"
            payload = b"".join(struct.pack("<6f", *record) for record in records)
            path.write_bytes(payload)
            means, digest = qi.activation_dump_means(
                path, expected_records=2, spec=spec, winsorization_quantile=1.0
            )
            self.assertEqual(means, [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            path.write_bytes(payload + b"\0")
            with self.assertRaisesRegex(qi.InterventionError, "expected exactly"):
                qi.activation_dump_means(
                    path, expected_records=2, spec=spec,
                    winsorization_quantile=1.0,
                )

    def test_corpus_content_overlap_is_rejected_even_with_different_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            good, bad, held = root / "good.jsonl", root / "bad.jsonl", root / "held.jsonl"
            write_corpus(good, [("g1", "same content")])
            write_corpus(bad, [("b1", "different")])
            write_corpus(held, [("h1", "same content")])
            evidence = [
                qi.scan_corpus(path, path.stem, max_records=10, max_line_bytes=4096)
                for path in (good, bad, held)
            ]
            with self.assertRaisesRegex(qi.InterventionError, "good/held-out"):
                qi.require_disjoint_corpora(*evidence)

    def test_dump_backend_generates_quantizer_consumable_per_layer_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            good, bad, held = root / "good.jsonl", root / "bad.jsonl", root / "held.jsonl"
            write_corpus(good, [("g1", "Give a useful safe answer.")])
            write_corpus(bad, [("b1", "Refuse without helping.")])
            write_corpus(held, [("h1", "Held out evaluation prompt.")])

            # Good and bad differ along dimension 1; dimension 0 gives the
            # orthogonalization path a non-degenerate control mean.
            good_row = [0.0] * (qi.QWEN_SPEC.layer_count * qi.QWEN_SPEC.hidden_size)
            bad_row = [0.0] * len(good_row)
            for layer in range(qi.QWEN_SPEC.layer_count):
                offset = layer * qi.QWEN_SPEC.hidden_size
                good_row[offset] = 1.0
                bad_row[offset] = 1.0
                bad_row[offset + 1] = 2.0 + layer / 100.0
            good_dump, bad_dump = root / "good.f32", root / "bad.f32"
            good_dump.write_bytes(struct.pack(f"<{len(good_row)}f", *good_row))
            bad_dump.write_bytes(struct.pack(f"<{len(bad_row)}f", *bad_row))
            output = root / "qwen-intervention-manifest.json"
            profile = ROOT / "share/release_profiles/qwen3.8-flash-next-rocmi4-strix-halo.json"
            completed = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts/qwen_intervention.py"),
                    "--profile", str(profile),
                    "--good-corpus", str(good),
                    "--bad-corpus", str(bad),
                    "--held-out-corpus", str(held),
                    "--good-activations", str(good_dump),
                    "--bad-activations", str(bad_dump),
                    "--activation-artifact-sha256", "a" * 64,
                    "--scale", "1.0",
                    "--output", str(output),
                ],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["directions"]), 48)
            self.assertEqual(len(manifest["targets"]), 33)
            self.assertEqual(
                manifest["extraction"]["semantic_capture_point"],
                "decoder_layer.residual_writer.output",
            )
            self.assertEqual(manifest["extraction"]["layer_policy"], "band-10-42")
            self.assertEqual(manifest["extraction"]["policy_evidence"], {
                "source_revision": "a3c6a728510f91394e991504951ac316cd3a89af",
                "deepseek_reference_band": "10-42",
                "qwen_status": "exploratory_transfer_hypothesis",
            })
            self.assertIn(
                "layer-00-residual-writer-output-r1",
                {row["id"] for row in manifest["directions"]},
            )
            self.assertTrue(all(len(row["values"]) == 2560 for row in manifest["directions"]))
            self.assertEqual(
                manifest["extraction"]["activation_evidence"]["backend"],
                "ember_qwen_runtime_f32_dump",
            )
            self.assertFalse(manifest["extraction"]["hidden_states_api_used"])
            release_profile, _inventory, _path = qwen_quantize.validate_profile(profile)
            _validated, evidence = qwen_quantize.validate_intervention_manifest(
                output, release_profile
            )
            self.assertEqual(evidence["direction_count"], 48)
            self.assertEqual(evidence["target_count"], 33)

            for field, value, expected in (
                ("qwen_status", "established_qwen_policy", "residual-writer outputs"),
                ("layer_policy", "upper-24", "exact exploratory layer policy"),
            ):
                with self.subTest(field=field):
                    tampered = json.loads(json.dumps(manifest))
                    if field == "layer_policy":
                        tampered["extraction"][field] = value
                    else:
                        tampered["extraction"]["policy_evidence"][field] = value
                    tampered_path = root / f"tampered-{field}.json"
                    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
                    with self.assertRaisesRegex(qwen_quantize.PipelineError, expected):
                        qwen_quantize.validate_intervention_manifest(
                            tampered_path, release_profile
                        )


if __name__ == "__main__":
    unittest.main()
