#!/usr/bin/env python3
"""GPU-free tests for the pinned Qwen conversion/quantization orchestrator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_quantize.py"
SPEC = importlib.util.spec_from_file_location("qwen_quantize", SCRIPT)
assert SPEC and SPEC.loader
qwen_quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qwen_quantize)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path: Path) -> str:
    value = path.read_bytes()
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value, usedforsecurity=False).hexdigest()


TEST_INTERVENTION = {
    "kind": "directional_ablation",
    "application_stage": "pre_quantization_encoding",
    "manifest_sha256": "8" * 64,
    "target_names_sha256": hashlib.sha256(b"blk.0.ssm_out.weight").hexdigest(),
    "target_count": 1,
}


def git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout.strip()


def init_repo(directory: Path) -> None:
    directory.mkdir()
    git(directory, "init", "-q")
    git(directory, "config", "user.email", "test@example.invalid")
    git(directory, "config", "user.name", "Qwen Test")


def commit_all(directory: Path, message: str) -> str:
    git(directory, "add", ".")
    git(directory, "commit", "-q", "-m", message)
    return git(directory, "rev-parse", "HEAD")


def make_safetensors(path: Path) -> None:
    values = {
        "model.ple_embedding.layer_multipliers": [1, 24_000_000_000_001],
        "model.ple_embedding.ngram_heads_offsets": [3, 5],
        "model.ple_embedding.ngram_heads_vocab_sizes": [7, 11],
    }
    offset = 0
    header = {}
    payload = bytearray()
    for name, items in values.items():
        raw = struct.pack("<" + "q" * len(items), *items)
        header[name] = {"dtype": "I64", "shape": [len(items)], "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def pack_string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def pack_value(value_type: int, value, subtype: int | None = None) -> bytes:
    formats = {2: "H", 4: "I", 5: "i", 8: None, 10: "Q"}
    if value_type == 8:
        return pack_string(value)
    if value_type == 9:
        assert subtype is not None
        return struct.pack("<IQ", subtype, len(value)) + b"".join(pack_value(subtype, item) for item in value)
    return struct.pack("<" + formats[value_type], value)


def make_gguf(
    path: Path, *, split_no: int, split_count: int, tensor_name: str,
    tensor_type: int, architecture: str = "qwen4exp",
    layer_multipliers: list[int] | None = None,
    intervention: dict | None = None,
) -> None:
    layer_multipliers = layer_multipliers or [1, 24_000_000_000_001]
    metadata = [
        ("general.architecture", 8, architecture, None),
        ("qwen4exp.ple.layer_multipliers", 9, layer_multipliers, 10),
        ("qwen4exp.ple.head_offsets", 9, [3, 5], 10),
        ("qwen4exp.ple.head_vocab_sizes", 9, [7, 11], 10),
    ]
    if intervention is not None:
        metadata.extend([
            ("ember.intervention.kind", 8, intervention["kind"], None),
            ("ember.intervention.application_stage", 8, intervention["application_stage"], None),
            ("ember.intervention.manifest_sha256", 8, intervention["manifest_sha256"], None),
            ("ember.intervention.target_names_sha256", 8, intervention["target_names_sha256"], None),
            ("ember.intervention.target_count", 4, intervention["target_count"], None),
        ])
    if split_count > 1:
        metadata.extend([
            ("split.no", 2, split_no, None),
            ("split.count", 2, split_count, None),
            ("split.tensors.count", 5, split_count, None),
        ])
    value = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)))
    for key, value_type, item, subtype in metadata:
        value.extend(pack_string(key))
        value.extend(struct.pack("<I", value_type))
        value.extend(pack_value(value_type, item, subtype))
    value.extend(pack_string(tensor_name))
    value.extend(struct.pack("<IQQIQ", 2, 160, 1, tensor_type, 0))
    path.write_bytes(value)


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.snapshot = root / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "LICENSE").write_text("fixture license\n", encoding="utf-8")
        (self.snapshot / "config.json").write_text(json.dumps({
            "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
        }), encoding="utf-8")
        make_safetensors(self.snapshot / "model-00001-of-00001.safetensors")
        self.intermediate_template = root / "intermediate-template.gguf"
        self.quant_template = root / "quant-template.gguf"
        make_gguf(
            self.intermediate_template, split_no=0, split_count=1,
            tensor_name="per_layer_token_embd.weight", tensor_type=30,
        )
        make_gguf(
            self.quant_template, split_no=0, split_count=1,
            tensor_name="per_layer_token_embd.weight", tensor_type=108,
        )
        self.intermediate_split = [
            root / "intermediate-split-00001-of-00002.gguf",
            root / "intermediate-split-00002-of-00002.gguf",
        ]
        self.quant_split = [
            root / "quant-split-00001-of-00002.gguf",
            root / "quant-split-00002-of-00002.gguf",
        ]
        make_gguf(
            self.intermediate_split[0], split_no=0, split_count=2,
            tensor_name="per_layer_token_embd.weight", tensor_type=30,
        )
        make_gguf(
            self.intermediate_split[1], split_no=1, split_count=2,
            tensor_name="blk.0.ffn_up.weight", tensor_type=30,
        )
        make_gguf(
            self.quant_split[0], split_no=0, split_count=2,
            tensor_name="per_layer_token_embd.weight", tensor_type=108,
        )
        make_gguf(
            self.quant_split[1], split_no=1, split_count=2,
            tensor_name="blk.0.ffn_up.weight", tensor_type=108,
        )

        self.llama = root / "llama"
        init_repo(self.llama)
        (self.llama / "conversion").mkdir()
        (self.llama / "convert_hf_to_gguf.py").write_text(
            "#!/usr/bin/python3\nimport pathlib,shutil,sys\n"
            "out=pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
            "if '--split-max-size' in sys.argv:\n"
            f" sources={[str(path) for path in self.intermediate_split]!r}\n"
            " for index,source in enumerate(sources,1):\n"
            "  shutil.copyfile(source,out.with_name(f'{out.stem}-{index:05d}-of-00002.gguf'))\n"
            "else:\n"
            f" shutil.copyfile({str(self.intermediate_template)!r},out)\n",
            encoding="utf-8",
        )
        (self.llama / "conversion" / "qwen4exp.py").write_text(
            "Qwen4ExpForConditionalGeneration = True\ndef _read_hash_constants(): pass\n", encoding="utf-8",
        )
        self.llama_base = commit_all(self.llama, "base")
        (self.llama / "rotated-kv.txt").write_text("mandatory fix\n", encoding="utf-8")
        self.gguf_splitter = self.llama / "llama-gguf-split"
        self.gguf_splitter.write_text(
            "#!/usr/bin/python3\nimport pathlib,shutil,subprocess,sys\n"
            "if sys.argv[1:] == ['--version']:\n"
            " print('commit ' + subprocess.check_output(['git','rev-parse','HEAD'],cwd=pathlib.Path(__file__).parent,text=True).strip())\n"
            f"else:\n sources={[str(path) for path in self.intermediate_split]!r}\n"
            " out=pathlib.Path(sys.argv[-1])\n"
            " for index,source in enumerate(sources,1):\n"
            "  shutil.copyfile(source,out.with_name(f'{out.stem}-{index:05d}-of-00002.gguf'))\n",
            encoding="utf-8",
        )
        self.gguf_splitter.chmod(self.gguf_splitter.stat().st_mode | stat.S_IXUSR)
        self.llama_head = commit_all(self.llama, "rotated kv")

        self.rocmfpx = root / "rocmfpx"
        init_repo(self.rocmfpx)
        (self.rocmfpx / "tools" / "quantize").mkdir(parents=True)
        (self.rocmfpx / "tools" / "quantize" / "quantize.cpp").write_text(
            "Q4_0_ROCMI4 arg_idx < argc && strncmp\n", encoding="utf-8",
        )
        self.rocm_revision = commit_all(self.rocmfpx, "rocmi4")

        self.ember = root / "ember"
        init_repo(self.ember)
        self.fail_quantizer = root / "fail-quantizer"
        self.quantizer_started = root / "quantizer-started"
        self.pause_success = root / "pause-success"
        self.success_written = root / "success-written"
        self.quantizer = self.ember / "ember-gguf-quantize"
        self.quantizer.write_text(
            "#!/usr/bin/python3\n"
            "import hashlib,json,pathlib,shutil,subprocess,sys,time\n"
            "if sys.argv[1:] == ['--build-info-json']:\n"
            f" print(json.dumps({{'tool':'ember-gguf-quantize','ember_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=pathlib.Path(__file__).parent,text=True).strip(),"
            f"'rocmfpx_revision':'{self.rocm_revision}','format':'Q4_0_ROCMI4','ggml_tensor_type':108,'intervention_manifest_schema':1}}))\n"
            "elif '--dry-size-json' in sys.argv:\n"
            " intervention={}\n"
            " if '--intervention-manifest' in sys.argv:\n"
            "  manifest_path=pathlib.Path(sys.argv[sys.argv.index('--intervention-manifest') + 1])\n"
            "  manifest_bytes=manifest_path.read_bytes(); manifest=json.loads(manifest_bytes)\n"
            "  target_names=[item['tensor_name'] for item in manifest['targets']]\n"
            "  intervention={'intervention_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),"
            "'intervention_target_names_sha256':manifest['tensor_map']['target_names_sha256'],"
            "'intervention_target_count':len(target_names),'intervention_targets':target_names,"
            "'intervention_validated':True,'intervention_applied':False}\n"
            f" templates=[pathlib.Path(path) for path in {[str(path) for path in self.quant_split]!r}] if '--keep-split' in sys.argv else [pathlib.Path({str(self.quant_template)!r})]\n"
            " shard_bytes=[path.stat().st_size for path in templates]\n"
            " artifact=sum(shard_bytes)\n"
            " reserve=int(sys.argv[sys.argv.index('--runtime-reserve-bytes') + 1])\n"
            " budget=int(sys.argv[sys.argv.index('--device-budget-bytes') + 1])\n"
            " total=artifact+reserve\n"
            " print(json.dumps({'artifact_bytes':artifact,'shard_count':len(shard_bytes),'shard_bytes':shard_bytes,'runtime_reserve_bytes':reserve,"
            "'budget_bytes':budget,'total_bytes':total,'headroom_bytes':budget-total,"
            "'fits':total <= budget,**intervention}))\n"
            f"elif pathlib.Path({str(self.fail_quantizer)!r}).exists():\n"
            f" pathlib.Path({str(self.quantizer_started)!r}).write_text('started')\n"
            " time.sleep(0.5)\n"
            " sys.exit(9)\n"
            "else:\n"
            " out=pathlib.Path(sys.argv[-3])\n"
            " if '--keep-split' in sys.argv:\n"
            f"  sources={[str(path) for path in self.quant_split]!r}\n"
            "  for index,source in enumerate(sources,1):\n"
            "   shutil.copyfile(source,out.with_name(f'{out.stem}-{index:05d}-of-00002.gguf'))\n"
            " else:\n"
            f"  shutil.copyfile({str(self.quant_template)!r},out)\n"
            f" if pathlib.Path({str(self.pause_success)!r}).exists():\n"
            f"  pathlib.Path({str(self.success_written)!r}).write_text('written')\n"
            "  time.sleep(0.5)\n"
            " if '--intervention-manifest' in sys.argv:\n"
            "  manifest_path=pathlib.Path(sys.argv[sys.argv.index('--intervention-manifest') + 1])\n"
            "  manifest_bytes=manifest_path.read_bytes(); manifest=json.loads(manifest_bytes)\n"
            "  target_names=[item['tensor_name'] for item in manifest['targets']]\n"
            "  print(json.dumps({'intervention_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),"
            "'intervention_target_names_sha256':manifest['tensor_map']['target_names_sha256'],"
            "'intervention_target_count':len(target_names),'intervention_targets':target_names,"
            "'intervention_validated':True,'intervention_applied':True,"
            "'intervention_metrics':[{'tensor_name':target_names[0],'source_projection_l2':1.0,"
            "'stored_projection_l2':0.2,'stored_projection_ratio':0.2,"
            "'signed_projection_coefficient':-0.8,'relative_frobenius_delta':0.05,"
            "'row_norm_relative_rmse':0.001,'row_norm_relative_max':0.002}]}))\n",
            encoding="utf-8",
        )
        self.quantizer.chmod(self.quantizer.stat().st_mode | stat.S_IXUSR)
        self.ember_revision = commit_all(self.ember, "generic quantizer")

        revision = "a" * 40
        files = []
        for path in sorted(self.snapshot.iterdir()):
            files.append({"path": path.name, "size": path.stat().st_size, "git_blob": git_blob(path)})
        canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
        self.inventory = root / "inventory.json"
        self.inventory.write_text(json.dumps({
            "schema_version": 1, "repo_id": "fixture/qwen", "revision": revision,
            "file_count": len(files), "total_bytes": sum(item["size"] for item in files),
            "inventory_sha256": hashlib.sha256(canonical).hexdigest(), "files": files,
        }), encoding="utf-8")
        base_profile = json.loads((ROOT / "share" / "release_profiles" / "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text())
        base_profile["source"].update({
            "repo_id": "fixture/qwen", "revision": revision,
            "weight_bytes": (self.snapshot / "model-00001-of-00001.safetensors").stat().st_size,
            "snapshot_inventory": "inventory.json", "snapshot_inventory_sha256": sha256(self.inventory),
        })
        base_profile["source"]["license"]["sha256"] = sha256(self.snapshot / "LICENSE")
        base_profile["conversion"]["base_revision"] = self.llama_base
        base_profile["conversion"]["revision"] = self.llama_head
        base_profile["quantizer"]["revision"] = self.rocm_revision
        base_profile["quantization"]["source_provenance"] = {
            "qwen4exp": self.llama_head, "rocmi4": self.rocm_revision,
        }
        self.profile = root / "profile.json"
        self.profile.write_text(json.dumps(base_profile), encoding="utf-8")
        direction_values = [1.0]
        direction_sha = hashlib.sha256(struct.pack("<f", *direction_values)).hexdigest()
        target_names = ["blk.0.ssm_out.weight"]
        target_names_sha = hashlib.sha256("\n".join(sorted(target_names)).encode()).hexdigest()
        self.intervention_manifest = root / "intervention.json"
        self.intervention_manifest.write_text(json.dumps({
            "schema_version": 1,
            "kind": "directional_ablation",
            "status": "complete",
            "weight_intervention": True,
            "prompt_only": False,
            "application_stage": "pre_quantization_encoding",
            "source": {
                "repo_id": base_profile["source"]["repo_id"],
                "revision": base_profile["source"]["revision"],
                "snapshot_inventory_sha256": base_profile["source"]["snapshot_inventory_sha256"],
            },
            "tooling": {
                "otheru_quant_pipeline": base_profile["intervention"]["otheru_pipeline"],
                "upstream_heretic": base_profile["intervention"]["upstream_heretic"],
            },
            "corpora": [{
                "id": "fixture-refusal-pairs",
                "role": "direction_extraction",
                "sha256": "7" * 64,
                "record_count": 2,
                "held_out_evaluation_overlap_count": 0,
            }],
            "directions": [{
                "id": "refusal-r1", "dtype": "F32",
                "values": direction_values, "sha256": direction_sha,
            }],
            "targets": [{
                "tensor_name": target_names[0], "direction_id": "refusal-r1",
                "scale": 1.0, "normalization": "row_norm_preserve",
                "expected_shape": [160, 1],
            }],
            "tensor_map": {
                "kind": "exact_tensor_names", "target_count": 1,
                "target_names_sha256": target_names_sha,
            },
        }), encoding="utf-8")
        intervention_evidence = {
            "kind": "directional_ablation",
            "application_stage": "pre_quantization_encoding",
            "manifest_sha256": sha256(self.intervention_manifest),
            "target_names_sha256": target_names_sha,
            "target_count": 1,
        }
        make_gguf(
            self.quant_template, split_no=0, split_count=1,
            tensor_name="per_layer_token_embd.weight", tensor_type=108,
            intervention=intervention_evidence,
        )
        make_gguf(
            self.quant_split[0], split_no=0, split_count=2,
            tensor_name="per_layer_token_embd.weight", tensor_type=108,
            intervention=intervention_evidence,
        )
        make_gguf(
            self.quant_split[1], split_no=1, split_count=2,
            tensor_name="blk.0.ffn_up.weight", tensor_type=108,
            intervention=intervention_evidence,
        )
        self.revision = revision
        self.work = root / "work"

    def command(self) -> list[str]:
        return [
            sys.executable, str(SCRIPT), "--profile", str(self.profile),
            "--snapshot-dir", str(self.snapshot), "--snapshot-revision", self.revision,
            "--intervention-manifest", str(self.intervention_manifest),
            "--llama-cpp-dir", str(self.llama), "--rocmfpx-dir", str(self.rocmfpx),
            "--ember-dir", str(self.ember), "--ember-revision", self.ember_revision,
            "--quantizer", str(self.quantizer), "--work-dir", str(self.work),
            "--min-free-gib", "0", "--min-ram-gib", "0", "--threads", "7",
        ]


class QwenQuantizeTests(unittest.TestCase):
    def test_profile_selects_ram_floor_for_conversion_mode(self) -> None:
        args = qwen_quantize.parse_args([
            "--snapshot-dir", "/snapshot", "--snapshot-revision", "a" * 40,
            "--intervention-manifest", "/intervention.json",
            "--llama-cpp-dir", "/llama", "--rocmfpx-dir", "/rocmfpx",
            "--ember-dir", "/ember", "--ember-revision", "b" * 40,
            "--quantizer", "/quantizer", "--work-dir", "/work",
        ])
        self.assertIsNone(args.min_ram_gib)
        self.assertEqual(args.split_max_size, "48G")

    def test_default_is_validated_nonpublishing_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            result = subprocess.run(fixture.command(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((fixture.root / "work.plan.json").read_text())
            self.assertFalse(fixture.work.exists())
            self.assertEqual(record["mode"], "dry-run")
            self.assertEqual(record["status"], "planned")
            self.assertFalse(record["publishes"])
            self.assertFalse(record["credentials_accessed"])
            self.assertEqual(record["snapshot"]["files_verified"], 3)
            command = record["commands"]["quantize"]
            self.assertEqual(command[1], "--tensor-type")
            self.assertEqual(command[2], "^per_layer_token_embd\\.weight$=Q4_0_ROCMI4")
            self.assertEqual(command[3], "--intervention-manifest")
            self.assertEqual(command[4], str(fixture.intervention_manifest.resolve()))
            self.assertLess(command.index("--device-budget-bytes"), len(command) - 4)
            self.assertEqual(command[command.index("--device-budget-bytes") + 1], "137438953472")
            self.assertLess(command.index("--runtime-reserve-bytes"), len(command) - 4)
            self.assertEqual(command[command.index("--runtime-reserve-bytes") + 1], "34359738368")
            self.assertEqual(command[-2:], ["Q4_0_ROCMI4", "7"])
            self.assertIn("--dry-size-json", record["commands"]["quantize_preflight"])
            self.assertIn("--keep-split", command)
            self.assertEqual(record["commands"]["convert"][-2:], ["--split-max-size", "48G"])
            self.assertTrue(record["commands"]["quantizer_options_precede_positionals"])
            self.assertEqual(record["ple"]["source_dtype"], "I64")
            self.assertEqual(record["ple"]["gguf_metadata_dtype"], "ARRAY<UINT64>")
            self.assertTrue(record["intervention"]["weight_intervention"])
            self.assertFalse(record["intervention"]["prompt_only"])
            self.assertEqual(record["intervention"]["target_count"], 1)
            self.assertFalse(record["experiment"]["final_release_eligible"])
            self.assertEqual(
                record["experiment"]["eligibility_status"],
                "pending_measured_bakeoff_and_hardware_certification",
            )

    def test_explicit_stock_control_plan_is_unchanged_and_final_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            command = fixture.command()
            option = command.index("--intervention-manifest")
            command[option:option + 2] = ["--stock-control"]
            result = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((fixture.root / "work.plan.json").read_text())
            self.assertIsNone(record["intervention"])
            self.assertEqual(record["experiment"]["kind"], "stock_control")
            self.assertTrue(record["experiment"]["stock_weights_unchanged"])
            self.assertFalse(record["experiment"]["final_release_eligible"])
            quantize = record["commands"]["quantize"]
            self.assertNotIn("--intervention-manifest", quantize)
            self.assertIn("--tensor-type", quantize)

    def test_bounded_memory_plan_spills_then_splits_without_in_memory_registry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            command = fixture.command() + [
                "--bounded-memory-temp", "--gguf-splitter", str(fixture.gguf_splitter),
            ]
            result = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((fixture.root / "work.plan.json").read_text())
            convert = record["commands"]["convert"]
            self.assertIn("--use-temp-file", convert)
            self.assertNotIn("--split-max-size", convert)
            self.assertEqual(
                record["commands"]["split"][1:3], ["--split-max-size", "48G"]
            )
            self.assertFalse(record["conversion_memory"]["full_in_memory_tensor_registry"])
            self.assertEqual(
                record["conversion_memory"]["target_measurement_status"],
                "pending_peak_rss_and_wall_time",
            )
            self.assertIn("gguf_splitter_sha256", record["tools"])

    def test_bounded_memory_execute_removes_unsplit_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            command = fixture.command() + [
                "--bounded-memory-temp", "--gguf-splitter", str(fixture.gguf_splitter),
                "--execute",
            ]
            result = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(
                (fixture.work / "Qwen3.8-Flash-Next-BF16.unsplit.gguf").exists()
            )
            self.assertFalse((fixture.work / ".converter-tmp").exists())
            record = json.loads(
                (fixture.work / "qwen-quant-build-record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["conversion_memory"]["mode"], "bounded_temp_file_then_split")
            self.assertEqual(len(record["intermediate"]["shards"]), 2)

    def test_stock_control_verification_rejects_intervention_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            profile = json.loads(
                (ROOT / "share" / "release_profiles" /
                 "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text()
            )
            expected = {
                "ple_embedding.layer_multipliers": [1, 24_000_000_000_001],
                "ple_embedding.ngram_heads_offsets": [3, 5],
                "ple_embedding.ngram_heads_vocab_sizes": [7, 11],
            }
            clean = root / "clean.gguf"
            make_gguf(
                clean, split_no=0, split_count=1,
                tensor_name="per_layer_token_embd.weight", tensor_type=108,
            )
            qwen_quantize.verify_gguf_set(
                [clean], expected, quantized=True, profile=profile,
                stock_control=True,
            )
            edited = root / "edited.gguf"
            make_gguf(
                edited, split_no=0, split_count=1,
                tensor_name="per_layer_token_embd.weight", tensor_type=108,
                intervention=TEST_INTERVENTION,
            )
            with self.assertRaisesRegex(qwen_quantize.PipelineError, "stock-control"):
                qwen_quantize.verify_gguf_set(
                    [edited], expected, quantized=True, profile=profile,
                    stock_control=True,
                )

    def test_stock_control_executes_and_commits_clean_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            make_gguf(
                fixture.quant_split[0], split_no=0, split_count=2,
                tensor_name="per_layer_token_embd.weight", tensor_type=108,
            )
            make_gguf(
                fixture.quant_split[1], split_no=1, split_count=2,
                tensor_name="blk.0.ffn_up.weight", tensor_type=108,
            )
            command = fixture.command()
            option = command.index("--intervention-manifest")
            command[option:option + 2] = ["--stock-control"]
            result = subprocess.run(
                command + ["--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(
                (fixture.work / "qwen-quant-build-record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["experiment"]["kind"], "stock_control")
            self.assertFalse(record["experiment"]["final_release_eligible"])
            self.assertEqual(len(record["output"]["shards"]), 2)
            self.assertEqual(record["staging_transaction"]["evidence_promoted"], [])

    def test_dry_run_plan_does_not_block_execute_directory_commit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            planned = subprocess.run(
                fixture.command(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            self.assertFalse(fixture.work.exists())
            executed = subprocess.run(
                [*fixture.command(), "--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(executed.returncode, 0, executed.stderr)
            self.assertTrue((fixture.work / "qwen-quant-build-record.json").is_file())

    def test_snapshot_inventory_rejects_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            (fixture.snapshot / "unexpected.txt").write_text("no\n", encoding="utf-8")
            result = subprocess.run(fixture.command(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 2)
            self.assertIn("snapshot file inventory mismatch", result.stderr)

    def test_execute_runs_both_commands_and_completes_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            result = subprocess.run(
                [*fixture.command(), "--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((fixture.work / "qwen-quant-build-record.json").read_text())
            self.assertEqual(record["mode"], "execute")
            self.assertEqual(record["status"], "complete")
            self.assertTrue(record["memory_preflight"]["fits"])
            self.assertEqual(record["memory_preflight"]["budget_bytes"], 137438953472)
            self.assertEqual(record["memory_preflight"]["runtime_reserve_bytes"], 34359738368)
            self.assertEqual(
                record["memory_preflight"]["certification_host_memtotal_bytes"],
                134297894912,
            )
            self.assertEqual(
                record["memory_preflight"]["companion_artifact_fit_status"],
                "pending_mtp_mmproj_inventory",
            )
            self.assertEqual(record["memory_preflight"]["shard_count"], 2)
            self.assertEqual(
                record["memory_preflight"]["shard_bytes"],
                [item["size_bytes"] for item in record["output"]["shards"]],
            )
            self.assertEqual(record["output"]["tensor_type_counts"], {"108": 2})
            self.assertEqual(record["intermediate"]["tensor_names_sha256"], record["output"]["tensor_names_sha256"])
            self.assertTrue(record["intervention"]["quantizer_preflight"]["validated"])
            self.assertFalse(record["intervention"]["quantizer_preflight"]["applied"])
            self.assertTrue(record["intervention"]["quantizer_application"]["applied"])
            self.assertTrue(record["staging_transaction"]["same_filesystem"])
            self.assertTrue(record["staging_transaction"]["verified_before_promotion"])
            self.assertEqual(record["staging_transaction"]["boundary"], "atomic_directory")
            self.assertEqual(
                record["staging_transaction"]["commit_method"],
                "renameat2(RENAME_NOREPLACE)",
            )
            self.assertEqual(len(record["staging_transaction"]["promoted"]), 2)
            self.assertEqual(len(record["staging_transaction"]["evidence_promoted"]), 1)
            self.assertEqual(list(fixture.root.glob(".work.transaction-*")), [])
            self.assertTrue((fixture.work / "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00001-of-00002.gguf").is_file())
            self.assertTrue((fixture.work / "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00002-of-00002.gguf").is_file())
            self.assertEqual(
                (fixture.work / "qwen-intervention-manifest.json").read_bytes(),
                fixture.intervention_manifest.read_bytes(),
            )

    def test_rejects_prompt_only_or_unpinned_intervention_manifest(self) -> None:
        for mutation, expected in (("prompt", "not prompt-only"), ("tool", "not pinned")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                manifest = json.loads(fixture.intervention_manifest.read_text())
                if mutation == "prompt":
                    manifest["weight_intervention"] = False
                    manifest["prompt_only"] = True
                else:
                    manifest["tooling"]["upstream_heretic"]["revision"] = "0" * 40
                fixture.intervention_manifest.write_text(json.dumps(manifest))
                result = subprocess.run(
                    fixture.command(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_rejects_intervention_tensor_map_or_direction_digest_mismatch(self) -> None:
        for mutation, expected in (("map", "tensor-map"), ("direction", "direction SHA-256")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                manifest = json.loads(fixture.intervention_manifest.read_text())
                if mutation == "map":
                    manifest["tensor_map"]["target_count"] = 0
                else:
                    manifest["directions"][0]["sha256"] = "0" * 64
                fixture.intervention_manifest.write_text(json.dumps(manifest))
                result = subprocess.run(
                    fixture.command(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_rejects_wrong_residual_writer_or_column_matched_direction(self) -> None:
        for mutation, expected in (("writer", "residual writer"), ("columns", "ne0 columns")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                manifest = json.loads(fixture.intervention_manifest.read_text())
                if mutation == "writer":
                    manifest["targets"][0]["tensor_name"] = "blk.0.attn_output.weight"
                else:
                    manifest["targets"][0]["expected_shape"] = [1, 160]
                fixture.intervention_manifest.write_text(json.dumps(manifest))
                result = subprocess.run(
                    fixture.command(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_execute_rerun_preserves_completed_outputs_and_record_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            command = [*fixture.command(), "--execute"]
            first = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            record_path = fixture.work / "qwen-quant-build-record.json"
            outputs = sorted(fixture.work.glob("*.gguf"))
            before = {path: path.read_bytes() for path in [record_path, *outputs]}

            second = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(second.returncode, 2)
            self.assertIn("refusing to overwrite existing transaction directory", second.stderr)
            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_split_gguf_verification_checks_ple_and_type_108(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            make_gguf(first, split_no=0, split_count=2, tensor_name="per_layer_token_embd.weight", tensor_type=108, intervention=TEST_INTERVENTION)
            make_gguf(second, split_no=1, split_count=2, tensor_name="blk.0.ffn_up.weight", tensor_type=108, intervention=TEST_INTERVENTION)
            profile = json.loads((ROOT / "share" / "release_profiles" / "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text())
            expected = {
                "ple_embedding.layer_multipliers": [1, 24_000_000_000_001],
                "ple_embedding.ngram_heads_offsets": [3, 5],
                "ple_embedding.ngram_heads_vocab_sizes": [7, 11],
            }
            result = qwen_quantize.verify_gguf_set([first, second], expected, quantized=True, profile=profile, intervention=TEST_INTERVENTION)
            self.assertEqual(result["tensor_count"], 2)
            self.assertEqual(result["tensor_type_counts"], {"108": 2})
            self.assertEqual(len(result["shards"]), 2)

    def test_split_verification_rejects_later_shard_architecture_or_ple_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            make_gguf(first, split_no=0, split_count=2,
                      tensor_name="per_layer_token_embd.weight", tensor_type=108,
                      intervention=TEST_INTERVENTION)
            profile = json.loads(
                (ROOT / "share" / "release_profiles" /
                 "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text()
            )
            expected = {
                "ple_embedding.layer_multipliers": [1, 24_000_000_000_001],
                "ple_embedding.ngram_heads_offsets": [3, 5],
                "ple_embedding.ngram_heads_vocab_sizes": [7, 11],
            }
            make_gguf(second, split_no=1, split_count=2,
                      tensor_name="blk.0.ffn_up.weight", tensor_type=108,
                      architecture="wrong", intervention=TEST_INTERVENTION)
            with self.assertRaisesRegex(qwen_quantize.PipelineError, "shard 2 architecture"):
                qwen_quantize.verify_gguf_set(
                    [first, second], expected, quantized=True, profile=profile,
                    intervention=TEST_INTERVENTION,
                )

            make_gguf(second, split_no=1, split_count=2,
                      tensor_name="blk.0.ffn_up.weight", tensor_type=108,
                      layer_multipliers=[1, 2], intervention=TEST_INTERVENTION)
            with self.assertRaisesRegex(qwen_quantize.PipelineError, "shard 2 PLE metadata"):
                qwen_quantize.verify_gguf_set(
                    [first, second], expected, quantized=True, profile=profile,
                    intervention=TEST_INTERVENTION,
                )

    def test_execute_rolls_back_final_shards_after_semantic_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            make_gguf(
                fixture.quant_split[1], split_no=1, split_count=2,
                tensor_name="blk.0.ffn_up.weight", tensor_type=108,
                architecture="wrong",
            )
            result = subprocess.run(
                [*fixture.command(), "--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("shard 2 architecture", result.stderr)
            outputs = list(fixture.work.glob(
                "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-?????-of-?????.gguf"
            ))
            self.assertEqual(outputs, [])
            self.assertFalse(fixture.work.exists())
            self.assertEqual(list(fixture.root.glob(".work.transaction-*")), [])

    def test_failed_quantizer_does_not_delete_concurrent_foreign_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            fixture.fail_quantizer.write_text("fail\n", encoding="utf-8")
            process = subprocess.Popen(
                [*fixture.command(), "--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while not fixture.quantizer_started.exists() and process.poll() is None:
                if time.monotonic() >= deadline:
                    process.kill()
                    process.communicate()
                    self.fail("failing quantizer did not reach its race window")
                time.sleep(0.005)
            fixture.work.mkdir()
            foreign = fixture.work / (
                "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00002-of-00002.gguf"
            )
            foreign.write_bytes(b"foreign concurrent sentinel")
            _, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 2)
            self.assertIn("command failed (9)", stderr)
            self.assertEqual(foreign.read_bytes(), b"foreign concurrent sentinel")
            self.assertFalse((fixture.work / "qwen-quant-build-record.json").exists())

    def test_successful_quantizer_handoff_preserves_foreign_final(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            fixture.pause_success.write_text("pause\n", encoding="utf-8")
            process = subprocess.Popen(
                [*fixture.command(), "--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while not fixture.success_written.exists() and process.poll() is None:
                if time.monotonic() >= deadline:
                    process.kill()
                    process.communicate()
                    self.fail("successful quantizer did not reach its handoff window")
                time.sleep(0.005)
            fixture.work.mkdir()
            foreign = fixture.work / (
                "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00002-of-00002.gguf"
            )
            sentinel = b"foreign final during successful handoff"
            foreign.write_bytes(sentinel)
            _, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 2)
            self.assertIn("refusing to overwrite existing transaction directory", stderr)
            self.assertEqual(foreign.read_bytes(), sentinel)
            self.assertFalse((fixture.work / (
                "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00001-of-00002.gguf"
            )).exists())
            self.assertEqual(list(fixture.root.glob(".work.transaction-*")), [])

    def test_atomic_directory_publication_never_touches_foreign_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            staged = root / "private"
            final = root / "committed"
            staged.mkdir()
            (staged / "model.gguf").write_bytes(b"owned")
            final.mkdir()
            (final / "sentinel").write_bytes(b"foreign")
            with self.assertRaisesRegex(qwen_quantize.PipelineError, "transaction directory"):
                qwen_quantize.rename_directory_noreplace(staged, final)
            self.assertEqual((final / "sentinel").read_bytes(), b"foreign")
            self.assertEqual((staged / "model.gguf").read_bytes(), b"owned")

    def test_private_record_update_has_no_identity_check_replace_window(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "record.json"
            qwen_quantize.write_json_atomic(path, {"status": "planned"}, create=True)
            with mock.patch.object(qwen_quantize.os, "lstat", side_effect=AssertionError):
                qwen_quantize.write_json_atomic(path, {"status": "complete"}, create=False)
            self.assertEqual(json.loads(path.read_text())["status"], "complete")

    def test_failed_directory_commit_publishes_neither_shards_nor_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            args = qwen_quantize.parse_args([*fixture.command()[2:], "--execute"])
            with mock.patch.object(
                qwen_quantize, "rename_directory_noreplace",
                side_effect=qwen_quantize.PipelineError("injected commit failure"),
            ):
                with self.assertRaisesRegex(qwen_quantize.PipelineError, "commit failure"):
                    qwen_quantize.orchestrate(args)
            self.assertFalse(fixture.work.exists())
            self.assertEqual(list(fixture.root.glob(".work.transaction-*")), [])

    def test_failed_manifest_transaction_copy_leaves_no_private_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            args = qwen_quantize.parse_args([*fixture.command()[2:], "--execute"])
            with mock.patch.object(
                qwen_quantize, "copy_verified_input",
                side_effect=qwen_quantize.PipelineError("injected manifest copy failure"),
            ):
                with self.assertRaisesRegex(qwen_quantize.PipelineError, "manifest copy failure"):
                    qwen_quantize.orchestrate(args)
            self.assertFalse(fixture.work.exists())
            self.assertEqual(list(fixture.root.glob(".work.transaction-*")), [])

    def test_dangling_work_directory_symlink_is_a_no_clobber_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            missing = fixture.root / "foreign-missing-target"
            fixture.work.symlink_to(missing, target_is_directory=True)
            result = subprocess.run(
                [*fixture.command(), "--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("existing transaction directory", result.stderr)
            self.assertTrue(fixture.work.is_symlink())
            self.assertFalse(missing.exists())

    def test_dangling_dry_run_record_is_a_no_clobber_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            plan = fixture.root / "work.plan.json"
            missing = fixture.root / "foreign-missing-plan"
            plan.symlink_to(missing)
            result = subprocess.run(
                fixture.command(), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("existing build record", result.stderr)
            self.assertTrue(plan.is_symlink())
            self.assertFalse(missing.exists())

    def test_memory_preflight_rejects_artifact_plus_reserve_over_budget(self) -> None:
        profile = json.loads(
            (ROOT / "share" / "release_profiles" /
             "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text()
        )
        gate = profile["quantization"]["native_262k_memory_gate"]
        artifact = gate["device_budget_bytes"] - gate["runtime_reserve_bytes"] + 1
        with self.assertRaisesRegex(qwen_quantize.PipelineError, "does not fit"):
            qwen_quantize.validate_memory_preflight({
                "artifact_bytes": artifact,
                "shard_count": 2,
                "shard_bytes": [artifact // 2, artifact - artifact // 2],
                "runtime_reserve_bytes": gate["runtime_reserve_bytes"],
                "budget_bytes": gate["device_budget_bytes"],
                "total_bytes": artifact + gate["runtime_reserve_bytes"],
                "headroom_bytes": -1,
                "fits": False,
            }, profile)

    def test_memory_preflight_uses_real_otheru_memtotal_not_nominal_128g(self) -> None:
        profile = json.loads(
            (ROOT / "share" / "release_profiles" /
             "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text()
        )
        gate = profile["quantization"]["native_262k_memory_gate"]
        artifact = (
            gate["certification_host_memtotal_bytes"]
            - gate["runtime_reserve_bytes"] + 1
        )
        total = artifact + gate["runtime_reserve_bytes"]
        self.assertLess(total, gate["device_budget_bytes"])
        with self.assertRaisesRegex(qwen_quantize.PipelineError, "certification-host MemTotal"):
            qwen_quantize.validate_memory_preflight({
                "artifact_bytes": artifact,
                "shard_count": 1,
                "shard_bytes": [artifact],
                "runtime_reserve_bytes": gate["runtime_reserve_bytes"],
                "budget_bytes": gate["device_budget_bytes"],
                "total_bytes": total,
                "headroom_bytes": gate["device_budget_bytes"] - total,
                "fits": True,
            }, profile)

    def test_application_report_requires_finite_metrics_in_target_order(self) -> None:
        evidence = {
            "manifest_sha256": "1" * 64,
            "target_names_sha256": "2" * 64,
            "target_count": 1,
            "targets": ["blk.0.ssm_out.weight"],
        }
        base = {
            "intervention_manifest_sha256": evidence["manifest_sha256"],
            "intervention_target_names_sha256": evidence["target_names_sha256"],
            "intervention_target_count": 1,
            "intervention_targets": evidence["targets"],
            "intervention_validated": True,
            "intervention_applied": True,
            "intervention_metrics": [{
                "tensor_name": "blk.0.ssm_out.weight",
                "source_projection_l2": 1.0,
                "stored_projection_l2": 0.2,
                "stored_projection_ratio": 0.2,
                "signed_projection_coefficient": -0.8,
                "relative_frobenius_delta": 0.05,
                "row_norm_relative_rmse": 0.001,
                "row_norm_relative_max": 0.002,
            }],
        }
        validated = qwen_quantize.validate_intervention_report(base, evidence, applied=True)
        self.assertEqual(validated["metrics"][0]["tensor_name"], evidence["targets"][0])
        base["intervention_metrics"][0]["stored_projection_ratio"] = float("nan")
        with self.assertRaisesRegex(qwen_quantize.PipelineError, "finite and non-negative"):
            qwen_quantize.validate_intervention_report(base, evidence, applied=True)


if __name__ == "__main__":
    unittest.main()
