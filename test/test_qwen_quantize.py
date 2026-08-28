#!/usr/bin/env python3
"""GPU-free tests for the pinned Qwen conversion/quantization orchestrator."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
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


def make_mtp_companion(path: Path, source_revision: str, matrix_contract: str) -> None:
    def string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = (
        ("general.architecture", 8, "qwen4exp-mtp"),
        ("qwen4exp-mtp.source_revision", 8, source_revision),
        ("qwen4exp-mtp.shared_main_weights", 7, True),
        ("ember.mtp.matrix_quant_contract", 8, matrix_contract),
    )
    names = sorted(qwen_quantize.MTP_QUANTIZED_MATRIX_NAMES |
                   qwen_quantize.MTP_BF16_TENSOR_NAMES)
    output = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(names), len(metadata)))
    for key, kind, value in metadata:
        output += string(key) + struct.pack("<I", kind)
        output += string(value) if kind == 8 else struct.pack("<?", value)
    matrix_type = qwen_quantize.SUPPORTED_TENSOR_FORMATS[matrix_contract]
    for offset, name in enumerate(names):
        tensor_type = (matrix_type if name in qwen_quantize.MTP_QUANTIZED_MATRIX_NAMES
                       else 30)
        output += string(name) + struct.pack("<IQIQ", 1, 32, tensor_type, offset * 32)
    output += b"\0" * ((-len(output)) % 32)
    output += b"\0" * (len(names) * 32)
    path.write_bytes(output)


def make_mmproj_companion(path: Path, mutation: str | None = None) -> None:
    def string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = (
        ("general.architecture", 8, "clip"),
        ("general.file_type", 4, 32),
        ("clip.projector_type", 8, "qwen3vl_merger"),
        ("clip.has_vision_encoder", 7, True),
        ("clip.vision.projection_dim", 4, 2560),
        ("clip.vision.spatial_merge_size", 4, 2),
    )
    tensors = [dict(name=row["name"], shape=list(row["shape"]))
               for row in qwen_quantize.vision_inventory.load_contract()["tensors"]]
    if mutation == "missing":
        tensors.pop()
    elif mutation == "duplicate":
        tensors[-1]["name"] = tensors[0]["name"]
    elif mutation == "wrong_shape":
        tensors[0]["shape"][0] += 1
    elif mutation is not None:
        raise ValueError(f"unknown mmproj fixture mutation: {mutation}")
    output = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        output += string(key) + struct.pack("<I", kind)
        if kind == 8:
            output += string(value)
        elif kind == 7:
            output += struct.pack("<?", value)
        else:
            output += struct.pack("<I", value)
    for tensor in tensors:
        output += string(tensor["name"]) + struct.pack("<I", len(tensor["shape"]))
        output += struct.pack("<" + "Q" * len(tensor["shape"]), *tensor["shape"])
        output += struct.pack("<IQ", 30, 0)
    output += b"\0" * ((-len(output)) % 32) + b"\0"
    path.write_bytes(output)


def make_vision_vocab_companion(path: Path, *, with_tensor: bool = False) -> None:
    def string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = (
        ("general.architecture", 8, "qwen4exp", None),
        ("qwen4exp.embedding_length", 4, 2560, None),
        ("tokenizer.ggml.model", 8, "gpt2", None),
        ("tokenizer.ggml.tokens", 9, ["a", "b", "<vision>"], 8),
        ("tokenizer.ggml.token_type", 9, [1, 1, 3], 4),
    )
    tensor_count = 1 if with_tensor else 0
    output = bytearray(b"GGUF" + struct.pack("<IQQ", 3, tensor_count, len(metadata)))
    for key, kind, value, subtype in metadata:
        output += string(key) + struct.pack("<I", kind)
        if kind == 8:
            output += string(value)
        elif kind == 4:
            output += struct.pack("<I", value)
        else:
            output += struct.pack("<IQ", subtype, len(value))
            for item in value:
                output += string(item) if subtype == 8 else struct.pack("<I", item)
    if with_tensor:
        output += string("token_embd.weight") + struct.pack("<IQIQ", 1, 32, 30, 0)
        output += b"\0" * ((-len(output)) % 32) + b"\0" * 64
    path.write_bytes(output)


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
    stock_control: bool = False,
    quantized: bool = False,
    full_metadata: bool | None = None,
    omit_metadata_keys: set[str] | None = None,
    metadata_overrides: dict[str, tuple[int, object, int | None]] | None = None,
) -> None:
    layer_multipliers = layer_multipliers or [1, 24_000_000_000_001]
    if full_metadata is None:
        full_metadata = split_count == 1 or split_no == 0
    metadata = []
    if full_metadata:
        metadata.extend([
            ("general.architecture", 8, architecture, None),
            ("qwen4exp.ple.layer_multipliers", 9, layer_multipliers, 10),
            ("qwen4exp.ple.head_offsets", 9, [3, 5], 10),
            ("qwen4exp.ple.head_vocab_sizes", 9, [7, 11], 10),
        ])
    if quantized:
        metadata.extend([
            ("general.quantization_version", 4, 2, None),
            ("general.file_type", 4, 118, None),
        ])
    if intervention is not None and quantized:
        metadata.extend([
            ("ember.intervention.kind", 8, intervention["kind"], None),
            ("ember.intervention.application_stage", 8, intervention["application_stage"], None),
            ("ember.intervention.manifest_sha256", 8, intervention["manifest_sha256"], None),
            ("ember.intervention.target_names_sha256", 8, intervention["target_names_sha256"], None),
            ("ember.intervention.target_count", 4, intervention["target_count"], None),
        ])
    elif stock_control and quantized:
        metadata.extend([
            ("ember.intervention.kind", 8, "none_control", None),
            ("ember.intervention.release_eligibility", 8,
             "control_only_requires_manifest_for_release", None),
        ])
    if split_count > 1:
        metadata.extend([
            ("split.no", 2, split_no, None),
            ("split.count", 2, split_count, None),
            ("split.tensors.count", 5, split_count, None),
        ])
    omitted = omit_metadata_keys or set()
    metadata = [row for row in metadata if row[0] not in omitted]
    overrides = metadata_overrides or {}
    metadata = [
        (row[0], *overrides[row[0]]) if row[0] in overrides else row
        for row in metadata
    ]
    metadata.extend((key, *row) for key, row in overrides.items()
                    if all(existing[0] != key for existing in metadata))
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
        splitter_source = self.llama / "tools" / "gguf-split" / "gguf-split.cpp"
        splitter_source.parent.mkdir(parents=True)
        splitter_source.write_text(
            "#include <algorithm>\n#include <cstddef>\n#include <vector>\n"
            "struct split_strategy {\n"
            " std::vector<unsigned char> read_buf;\n"
            " void write(std::size_t n_bytes) { read_buf.resize(n_bytes); }\n"
            " void copy_file_to_file(std::size_t len) {\n"
            "  if (read_buf.size() < len) { read_buf.resize(len); }\n"
            " }\n};\n",
            encoding="utf-8",
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
            # Pinned llama.cpp's llama_split_path appends the shard suffix to
            # the complete output prefix; it does not strip a .gguf suffix.
            "  shutil.copyfile(source,out.with_name(f'{out.name}-{index:05d}-of-00002.gguf'))\n",
            encoding="utf-8",
        )
        self.gguf_splitter.chmod(self.gguf_splitter.stat().st_mode | stat.S_IXUSR)
        self.llama_head = commit_all(self.llama, "rotated kv")
        qwen_converter = self.llama / "conversion" / "qwen4exp.py"
        qwen_converter.write_text(
            qwen_converter.read_text(encoding="utf-8")
            + "\n# flush dirty PLE mmap pages, then madvise MADV_DONTNEED\n",
            encoding="utf-8",
        )
        ple_patch_text = git(
            self.llama, "diff", "--binary", "--", "conversion/qwen4exp.py") + "\n"
        splitter_source.write_text(
            "#include <algorithm>\n#include <cstddef>\n#include <vector>\n"
            "struct split_strategy {\n"
            " static constexpr size_t COPY_BUFFER_SIZE = 16 * 1024 * 1024;\n"
            " std::vector<unsigned char> read_buf = std::vector<unsigned char>(COPY_BUFFER_SIZE);\n"
            " void write(std::size_t) {}\n"
            " void copy_file_to_file(std::size_t len) {\n"
            "  std::size_t copied = 0;\n"
            "  while (copied < len) {\n"
            "   const std::size_t chunk = std::min(read_buf.size(), len - copied);\n"
            "   copied += chunk;\n"
            "  }\n"
            " }\n};\n",
            encoding="utf-8",
        )
        splitter_patch_text = git(
            self.llama, "diff", "--binary", "--", "tools/gguf-split/gguf-split.cpp") + "\n"

        self.rocmfpx = root / "rocmfpx"
        init_repo(self.rocmfpx)
        (self.rocmfpx / "tools" / "quantize").mkdir(parents=True)
        (self.rocmfpx / "tools" / "quantize" / "quantize.cpp").write_text(
            "Q4_0_ROCMI4 Q3_0_ROCMFPX arg_idx < argc && strncmp\n",
            encoding="utf-8",
        )
        self.rocm_revision = commit_all(self.rocmfpx, "rocmi4")

        self.ember = root / "ember"
        init_repo(self.ember)
        self.ple_patch = self.ember / "patches" / "llama.cpp" / "qwen4exp-ple-cgroup-writeback.patch"
        self.ple_patch.parent.mkdir(parents=True)
        self.ple_patch.write_text(ple_patch_text, encoding="utf-8")
        self.splitter_patch = (
            self.ember / "patches" / "llama.cpp" / "gguf-split-bounded-copy.patch")
        self.splitter_patch.write_text(splitter_patch_text, encoding="utf-8")
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
            f"'rocmfpx_revision':'{self.rocm_revision}','format':'Q4_0_ROCMI4','ggml_tensor_type':108,"
            "'per_tensor_formats':['Q4_0_ROCMI4','Q6_K','Q4_0_ROCMFP4_FAST','Q3_0_ROCMFPX'],"
            "'intervention_manifest_schema':1}))\n"
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
        base_profile["conversion"]["bounded_memory"].update({
            "ple_cgroup_writeback_patch": "patches/llama.cpp/qwen4exp-ple-cgroup-writeback.patch",
            "ple_cgroup_writeback_patch_sha256": sha256(self.ple_patch),
            "patched_qwen4exp_sha256": sha256(qwen_converter),
            "gguf_split_bounded_copy_patch":
                "patches/llama.cpp/gguf-split-bounded-copy.patch",
            "gguf_split_bounded_copy_patch_sha256": sha256(self.splitter_patch),
            "patched_gguf_split_source_sha256": sha256(splitter_source),
            "gguf_split_copy_buffer_bytes": 16 * 1024 * 1024,
        })
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
                "extractor": {
                    "implementation": "ember-qwen-residual-writer-activation-extractor",
                    "schema_version": 2,
                },
            },
            "extraction": {
                "semantic_capture_point": "decoder_layer.residual_writer.output",
                "transformers_hook_module": (
                    "model.language_model.layers.N."
                    "{linear_attn.out_proj|self_attn.o_proj}"
                ),
                "transformers_hook_value": "forward_output[:,-1,:]",
                "hidden_states_api_used": False,
                "policy_evidence": {
                    "source_revision": "a3c6a728510f91394e991504951ac316cd3a89af",
                    "deepseek_reference_band": "10-42",
                    "qwen_status": "exploratory_transfer_hypothesis",
                },
            },
            "corpora": [{
                "id": "fixture-refusal-pairs",
                "class": "bad_target",
                "role": "direction_extraction",
                "sha256": "7" * 64,
                "record_count": 2,
                "held_out_evaluation_overlap_count": 0,
            }],
            "held_out_evaluation": {
                "id": "fixture-sweep-validation", "sha256": "6" * 64,
                "record_count": 2, "overlap_count": 0,
                "comparison": "canonical_text_chat_messages_sha256",
            },
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
            intervention=intervention_evidence, quantized=True,
        )
        make_gguf(
            self.quant_split[0], split_no=0, split_count=2,
            tensor_name="per_layer_token_embd.weight", tensor_type=108,
            intervention=intervention_evidence, quantized=True,
        )
        make_gguf(
            self.quant_split[1], split_no=1, split_count=2,
            tensor_name="blk.0.ffn_up.weight", tensor_type=108,
            intervention=intervention_evidence, quantized=True,
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

    def rocmi4_sweep_plan(
        self, *, scale: float = 1.0,
        schema_version: int = qwen_quantize.SELECTION_PLAN_SCHEMA_VERSION,
    ) -> Path:
        profile = json.loads(self.profile.read_text(encoding="utf-8"))
        manifest = json.loads(self.intervention_manifest.read_text(encoding="utf-8"))
        arm = qwen_quantize.validated_quantization_arms(profile)["rocmi4-control"]
        layers = {str(layer): 0.0 for layer in range(48)}
        layers["0"] = scale
        direction_basis = {
            "source": manifest["source"],
            "tooling": manifest["tooling"],
            "extraction": manifest["extraction"],
            "corpora": [{key: corpus.get(key) for key in (
                "class", "role", "sha256", "record_count")}
                        for corpus in manifest["corpora"]],
        }
        plan = self.root / f"sweep-plan-{scale}.json"
        plan.write_text(json.dumps({
            "schema_version": schema_version,
            "phase_scope": "selection",
            "status": "planned_unmeasured", "publication_allowed": False,
            "release_profile": {"path": str(self.profile.resolve()),
                                "sha256": sha256(self.profile)},
            "corpora": {"sweep-validation.jsonl": {
                "path": str(self.root / "sweep-validation.jsonl"),
                "sha256": manifest["held_out_evaluation"]["sha256"],
                "record_count": manifest["held_out_evaluation"]["record_count"],
            }},
            "sweep_configurations": [{
                "id": "lambda-1.00-all-48", "quantization_arm": "rocmi4-control",
                "profile_sha256": sha256(self.profile),
                "quantization_overrides_sha256": arm["per_tensor_overrides_sha256"],
                "runtime_mode": "exact_dequant", "final_release_eligible": False,
                "direction_basis": direction_basis, "layer_scales": layers,
            }],
        }), encoding="utf-8")
        return plan

    def companion_args(
        self, *, enable_mmproj: bool = False,
        mtp_matrix: str = "Q4_0_ROCMI4",
    ) -> list[str]:
        mtp = self.root / "Qwen3.8-Flash-Next-MTP.gguf"
        make_mtp_companion(mtp, self.revision, mtp_matrix)
        mtp_manifest = self.root / "Qwen3.8-Flash-Next-MTP.export.json"
        mtp_manifest.write_text(json.dumps({
            "schema": "ember.qwen4exp.mtp-gguf-export.v1",
            "source_revision": self.revision,
            "quantized_output": str(mtp),
            "quantized_bytes": mtp.stat().st_size,
            "quantized_sha256": sha256(mtp),
            "quantized_matrix_contract": mtp_matrix,
            "quantized_matrix_ggml_type": qwen_quantize.SUPPORTED_TENSOR_FORMATS[mtp_matrix],
            "quantized_matrix_tensor_count": len(qwen_quantize.MTP_QUANTIZED_MATRIX_NAMES),
            "quantized_matrix_tensors": sorted(qwen_quantize.MTP_QUANTIZED_MATRIX_NAMES),
            "quantizer_sha256": "9" * 64,
            "quantizer_build_info": {
                "tool": "ember-gguf-quantize", "ember_revision": self.ember_revision,
                "rocmfpx_revision": self.rocm_revision, "format": "Q4_0_ROCMI4",
                "ggml_tensor_type": 108,
                "per_tensor_formats": list(qwen_quantize.SUPPORTED_TENSOR_FORMATS),
            },
            "runtime_status": "loadable quantized companion",
        }), encoding="utf-8")
        rows = [{
            "role": "mtp", "enabled": True, "path": str(mtp),
            "size_bytes": mtp.stat().st_size, "sha256": sha256(mtp),
            "matrix_quant_contract": mtp_matrix,
            "export_manifest_path": str(mtp_manifest),
            "export_manifest_sha256": sha256(mtp_manifest),
        }]
        if enable_mmproj:
            mmproj = self.root / "Qwen3.8-Flash-Next-BF16-mmproj.gguf"
            make_mmproj_companion(mmproj)
            vocab = self.root / "Qwen3.8-Flash-Next-vocab-only.gguf"
            make_vision_vocab_companion(vocab)
            rows.append({
                "role": "vision_mmproj", "enabled": True, "path": str(mmproj),
                "size_bytes": mmproj.stat().st_size, "sha256": sha256(mmproj),
                "format": "BF16",
                "tensor_inventory_sha256":
                    qwen_quantize.vision_inventory.load_contract()[
                        "tensor_inventory_sha256"],
                "text_model": {
                    "path": str(vocab), "size_bytes": vocab.stat().st_size,
                    "sha256": sha256(vocab), "format": "GGUF_VOCAB_ONLY",
                    "metadata_sha256":
                        qwen_quantize.validate_qwen_vocab_only_gguf(vocab)[
                            "metadata_sha256"],
                },
            })
        else:
            rows.append({"role": "vision_mmproj", "enabled": False})
        companion_inventory = self.root / "companion-inventory.json"
        companion_inventory.write_text(json.dumps({
            "schema": qwen_quantize.COMPANION_INVENTORY_SCHEMA,
            "source": {
                "repo_id": "fixture/qwen", "revision": self.revision,
                "snapshot_inventory_sha256": sha256(self.inventory),
            },
            "companions": rows,
        }), encoding="utf-8")
        pages_limit = self.root / "pages_limit"
        pages_limit.write_text("32505856\n", encoding="utf-8")
        return [
            "--companion-inventory", str(companion_inventory),
            "--companion-inventory-sha256", sha256(companion_inventory),
            "--ttm-pages-limit-path", str(pages_limit),
        ]


class QwenQuantizeTests(unittest.TestCase):
    def test_credential_free_environment_has_deterministic_non_host_identity(self) -> None:
        with mock.patch.dict(
                os.environ, {"USER": "host-user", "LOGNAME": "host-login",
                             "HOME": "/host/home", "AWS_SECRET_ACCESS_KEY": "secret"}):
            environment = qwen_quantize.credential_free_env()
        self.assertEqual(environment["USER"], "ember-qwen")
        self.assertEqual(environment["LOGNAME"], "ember-qwen")
        self.assertNotIn("HOME", environment)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)

    def test_profile_requires_ordered_mmproj_and_vocab_artifact_contracts(self) -> None:
        profile_path = (
            ROOT / "share" / "release_profiles" /
            "qwen3.8-flash-next-rocmi4-strix-halo.json"
        )
        original = json.loads(profile_path.read_text(encoding="utf-8"))
        inventory_name = original["source"]["snapshot_inventory"]
        inventory_source = profile_path.parent / inventory_name
        for mutation in ("missing", "reordered"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw_tmp:
                directory = Path(raw_tmp)
                profile = json.loads(json.dumps(original))
                companions = profile["artifact"]["required_companion_artifacts"]
                if mutation == "missing":
                    companions.pop()
                else:
                    companions.reverse()
                candidate = directory / profile_path.name
                candidate.write_text(json.dumps(profile), encoding="utf-8")
                (directory / inventory_name).write_bytes(inventory_source.read_bytes())
                with self.assertRaisesRegex(
                        qwen_quantize.PipelineError,
                        "exact ordered BF16 mmproj and vocab-only"):
                    qwen_quantize.validate_profile(candidate)

    def test_profile_winner_order_is_pinned_performance_first(self) -> None:
        profile = json.loads((
            ROOT / "share" / "release_profiles" /
            "qwen3.8-flash-next-rocmi4-strix-halo.json"
        ).read_text(encoding="utf-8"))
        arms = qwen_quantize.validated_quantization_arms(profile)
        self.assertIn("rocmi4-control", arms)
        profile["quantization"]["performance_bakeoff"]["winner_order"] = [
            "quality_score_desc", "decode_median_tps_desc",
            "prefill_median_tps_desc", "id_asc",
        ]
        with self.assertRaisesRegex(qwen_quantize.PipelineError,
                                    "pinned override contract"):
            qwen_quantize.validated_quantization_arms(profile)

    def test_git_revision_scopes_safe_directory_without_home(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            checkout = Path(raw) / "checkout"
            init_repo(checkout)
            (checkout / "tracked").write_text("bound checkout\n", encoding="utf-8")
            expected = commit_all(checkout, "initial")
            with mock.patch.object(
                    qwen_quantize, "run_checked",
                    wraps=qwen_quantize.run_checked) as checked:
                actual = qwen_quantize.git_revision(checkout)
            self.assertEqual(actual, expected)
            command = checked.call_args.args[0]
            self.assertEqual(command[1:3], ["-c", f"safe.directory={checkout.resolve()}"])
            self.assertEqual(command[3:], ["-C", str(checkout.resolve()), "rev-parse", "HEAD"])

    def test_ember_revision_uses_image_cmake_cache_without_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ember = Path(raw) / "ember-image-root"
            cache = ember / "build-rocm" / "CMakeCache.txt"
            cache.parent.mkdir(parents=True)
            revision = "1" * 40
            cache.write_text(
                f"EMBER_CONFIGURED_GIT_HEAD:STRING={revision}\n", encoding="utf-8")
            actual, source, has_git = qwen_quantize.ember_revision_evidence(ember)
            self.assertEqual(actual, revision)
            self.assertEqual(source, "cmake_cache")
            self.assertFalse(has_git)

    def test_ember_revision_rejects_cache_source_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ember = Path(raw) / "ember-source"
            init_repo(ember)
            (ember / "tracked").write_text("source\n", encoding="utf-8")
            source_revision = commit_all(ember, "source")
            cache = ember / "build-rocm" / "CMakeCache.txt"
            cache.parent.mkdir(parents=True)
            cache.write_text(
                f"EMBER_CONFIGURED_GIT_HEAD:STRING={'0' * 40}\n", encoding="utf-8")
            self.assertNotEqual(source_revision, "0" * 40)
            with self.assertRaisesRegex(
                    qwen_quantize.PipelineError, "differs from its configured"):
                qwen_quantize.ember_revision_evidence(ember)

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
            self.assertEqual(
                record["quantization_recipe"]["id"], "profile-default-rocmi4")
            self.assertEqual(
                record["quantization_recipe"]["selection"],
                "validated_top_level_profile_default",
            )
            self.assertEqual(
                record["quantization_recipe"]["per_tensor_overrides"],
                ["^per_layer_token_embd\\.weight$=Q4_0_ROCMI4"],
            )
            self.assertTrue(record["quantization_recipe"]["ple_override_preserved"])
            self.assertRegex(
                record["quantization_recipe"]["per_tensor_overrides_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(record["ple"]["source_dtype"], "I64")
            self.assertEqual(record["ple"]["gguf_metadata_dtype"], "ARRAY<UINT64>")
            self.assertTrue(record["intervention"]["weight_intervention"])
            self.assertFalse(record["intervention"]["prompt_only"])
            self.assertEqual(record["intervention"]["target_count"], 1)
            self.assertFalse(record["experiment"]["final_release_eligible"])
            self.assertEqual(
                record["experiment"]["eligibility_status"],
                "pending_exact_companion_inventory_and_measured_bakeoff_and_hardware_certification",
            )
            self.assertEqual(record["companion_inventory"]["status"], "not_supplied")
            self.assertIsNone(record["companion_inventory"]["enabled_artifact_bytes"])
            self.assertFalse(record["companion_inventory"]["estimated_bytes_used"])

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

    def test_directional_sweep_may_use_only_plan_bound_rocmi4_control(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            plan = fixture.rocmi4_sweep_plan()
            result = subprocess.run([
                *fixture.command(), "--quantization-arm", "rocmi4-control",
                "--bakeoff-plan", str(plan), "--bakeoff-plan-sha256", sha256(plan),
            ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((fixture.root / "work.plan.json").read_text())
            self.assertEqual(record["quantization_recipe"]["id"], "rocmi4-control")
            self.assertEqual(record["experiment"]["kind"], "directional_ablation")
            self.assertFalse(record["experiment"]["stock_weights_unchanged"])
            self.assertEqual(record["sweep_authorization"]["status"],
                             "authorized_selection_sweep_control_encoding")
            self.assertEqual(record["sweep_authorization"]["configuration_id"],
                             "lambda-1.00-all-48")
            self.assertFalse(record["sweep_authorization"]["final_release_eligible"])

    def test_rocmi4_sweep_control_rejects_loose_or_mismatched_authority(self) -> None:
        cases = ("missing", "one-sided", "bad-digest", "legacy-schema",
                 "wrong-extraction", "wrong-scale", "stock", "other-arm")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                plan = fixture.rocmi4_sweep_plan(
                    scale=0.5 if case == "wrong-scale" else 1.0,
                    schema_version=(1 if case == "legacy-schema" else
                                    qwen_quantize.SELECTION_PLAN_SCHEMA_VERSION),
                )
                if case == "wrong-extraction":
                    value = json.loads(plan.read_text(encoding="utf-8"))
                    value["sweep_configurations"][0]["direction_basis"][
                        "extraction"]["hidden_states_api_used"] = True
                    plan.write_text(json.dumps(value), encoding="utf-8")
                command = [*fixture.command(), "--quantization-arm", "rocmi4-control"]
                if case == "one-sided":
                    command += ["--bakeoff-plan", str(plan)]
                elif case not in {"missing"}:
                    command += ["--bakeoff-plan", str(plan), "--bakeoff-plan-sha256",
                                "0" * 64 if case == "bad-digest" else sha256(plan)]
                if case == "stock":
                    option = command.index("--intervention-manifest")
                    command[option:option + 2] = ["--stock-control"]
                elif case == "other-arm":
                    option = command.index("--quantization-arm")
                    command[option + 1] = "rocmi4-q6k-embedding-head"
                result = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(result.returncode, 2)
                expected = {
                    "missing": "exact canonical bakeoff plan descriptor",
                    "one-sided": "required together",
                    "bad-digest": "SHA-256 mismatch",
                    "legacy-schema": "selection-only canonical plan",
                    "wrong-extraction": "exactly one canonical sweep configuration",
                    "wrong-scale": "exactly one canonical sweep configuration",
                    "stock": "applies only to non-stock",
                    "other-arm": "applies only to non-stock",
                }[case]
                self.assertIn(expected, result.stderr)

    def test_profile_q6_and_rocmfp4_arms_plumb_all_overrides_in_order(self) -> None:
        expected = {
            "rocmi4-q6k-embedding-head": [
                "^per_layer_token_embd\\.weight$=Q4_0_ROCMI4",
                "^token_embd\\.weight$=Q6_K",
                "^output\\.weight$=Q6_K",
            ],
            "rocmfp4-fast-matrix": [
                "^per_layer_token_embd\\.weight$=Q4_0_ROCMI4",
                *[pattern + "=Q4_0_ROCMFP4_FAST"
                  for pattern in qwen_quantize.ROCMFP4_FAST_MATRIX_PATTERNS],
            ],
            "rocmfp4-fast-matrix-q3-ple-q6k-embedding-head": [
                "^per_layer_token_embd\\.weight$=Q3_0_ROCMFPX",
                *[pattern + "=Q4_0_ROCMFP4_FAST"
                  for pattern in qwen_quantize.ROCMFP4_FAST_MATRIX_PATTERNS],
                "^token_embd\\.weight$=Q6_K",
                "^output\\.weight$=Q6_K",
            ],
        }
        for arm_id, overrides in expected.items():
            with self.subTest(arm=arm_id), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                result = subprocess.run(
                    [*fixture.command(), "--quantization-arm", arm_id],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads((fixture.root / "work.plan.json").read_text())
                self.assertEqual(record["quantization_recipe"]["id"], arm_id)
                self.assertEqual(
                    record["quantization_recipe"]["per_tensor_overrides"], overrides)
                command = record["commands"]["quantize"]
                observed = [command[index + 1] for index, value in enumerate(command[:-1])
                            if value == "--tensor-type"]
                self.assertEqual(observed, overrides)
                self.assertEqual(
                    record["commands"]["quantize_preflight"].count("--tensor-type"),
                    len(overrides),
                )

    def test_profile_arms_fail_closed_on_unsupported_or_unpinned_overrides(self) -> None:
        mutations = [
            ("rocmi4-q6k-embedding-head", 1, "^token_embd\\.weight$=Q5_K",
             "malformed or unaudited"),
            ("rocmfp4-fast-matrix", 0,
             "^per_layer_token_embd\\.weight$=Q4_0_ROCMFP4_FAST",
             "malformed or unaudited"),
            ("rocmfp4-fast-matrix-q3-ple-q6k-embedding-head", 0,
             "^per_layer_token_embd\\.weight$=Q4_0_ROCMI4",
             "malformed or unaudited"),
        ]
        for arm_id, index, replacement, message in mutations:
            with self.subTest(arm=arm_id), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                profile = json.loads(fixture.profile.read_text())
                arm = next(item for item in profile["quantization"]
                           ["performance_bakeoff"]["arms"] if item["id"] == arm_id)
                arm["per_tensor_overrides"][index] = replacement
                fixture.profile.write_text(json.dumps(profile))
                result = subprocess.run(
                    fixture.command(), text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_stock_control_rejects_mixed_quantization_arm(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            command = fixture.command()
            option = command.index("--intervention-manifest")
            command[option:option + 2] = ["--stock-control"]
            result = subprocess.run(
                [*command, "--quantization-arm", "rocmi4-q6k-embedding-head"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("stock control must use", result.stderr)

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
            self.assertEqual(
                record["commands"]["split"][-1],
                str((fixture.root / "work" / "Qwen3.8-Flash-Next-BF16").resolve()),
            )
            self.assertFalse(record["commands"]["split"][-1].endswith(".gguf"))
            self.assertFalse(record["conversion_memory"]["full_in_memory_tensor_registry"])
            self.assertEqual(
                record["conversion_memory"]["target_measurement_status"],
                "pending_patched_peak_rss_and_wall_time",
            )
            self.assertIn("gguf_splitter_sha256", record["tools"])
            self.assertEqual(
                record["tools"]["gguf_split_copy_buffer_bytes"], 16 * 1024 * 1024)
            self.assertEqual(
                record["conversion_memory"]["split_copy_buffer_bytes"],
                16 * 1024 * 1024,
            )

    def test_splitter_provenance_rejects_unbounded_copy_regression(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            splitter_source = (
                fixture.llama / "tools" / "gguf-split" / "gguf-split.cpp")
            splitter_source.write_text(
                splitter_source.read_text(encoding="utf-8")
                + "\n// regression: read_buf.resize(len);\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                fixture.command(), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("patched GGUF splitter source digest differs", result.stderr)

    def test_splitter_patch_digest_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            fixture.splitter_patch.write_text(
                fixture.splitter_patch.read_text(encoding="utf-8") + "# tampered\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                fixture.command(), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("bounded-copy patch is missing or changed", result.stderr)

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
            self.assertEqual(record["conversion_memory"]["gguf_writer_temp_cleanup"], {
                "policy": "exact_converter_private_tmp_residue_v3",
                "removed": [],
            })
            self.assertEqual(len(record["intermediate"]["shards"]), 2)

    def test_cleanup_accepts_only_exact_gguf_writer_spool_residue(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw) / "private-tmp"
            directory.mkdir(mode=0o700)
            residue = directory / "tmpa1_b2c3d"
            residue.write_bytes(b"already consumed writer spool")
            residue.chmod(0o600)
            rows = qwen_quantize.cleanup_gguf_writer_temp(directory)
            self.assertEqual(rows, [{
                "name": residue.name, "size_bytes": 29, "mode": 0o600,
            }])
            self.assertFalse(directory.exists())

    def test_cleanup_accepts_bounded_owner_torchinductor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw) / "private-tmp"
            directory.mkdir(mode=0o700)
            cache = directory / "torchinductor_root"
            cache.mkdir(mode=0o700)
            rows = qwen_quantize.cleanup_gguf_writer_temp(directory)
            self.assertEqual(rows[0]["name"], "torchinductor_root")
            self.assertEqual(rows[0]["kind"], "bounded_torchinductor_cache_tree")
            self.assertEqual(rows[0]["entries"], 1)
            self.assertEqual(rows[0]["size_bytes"], 0)
            self.assertRegex(rows[0]["inventory_sha256"], r"^[0-9a-f]{64}$")
            self.assertFalse(directory.exists())

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw) / "private-tmp"
            directory.mkdir(mode=0o700)
            cache = directory / "torchinductor_root"
            cache.mkdir(mode=0o700)
            nested = cache / "fxgraph" / "ab"
            nested.mkdir(parents=True, mode=0o700)
            artifact = nested / "compiled.so"
            artifact.write_bytes(b"pinned torch cache artifact")
            artifact.chmod(0o755)
            rows = qwen_quantize.cleanup_gguf_writer_temp(directory)
            self.assertEqual(rows[0]["entries"], 4)
            self.assertEqual(rows[0]["size_bytes"], 27)
            self.assertFalse(directory.exists())

    def test_cleanup_rejects_unsafe_torchinductor_descendants(self) -> None:
        for case in ("symlink", "hardlink", "peer-writable", "special"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                directory = root / "private-tmp"
                directory.mkdir(mode=0o700)
                cache = directory / "torchinductor_root"
                cache.mkdir(mode=0o700)
                entry = cache / "artifact"
                if case == "symlink":
                    entry.symlink_to(root / "outside")
                elif case == "special":
                    os.mkfifo(entry)
                else:
                    entry.write_bytes(b"cache")
                    entry.chmod(0o600)
                    if case == "hardlink":
                        os.link(entry, root / "second-link")
                    else:
                        entry.chmod(0o622)
                with self.assertRaisesRegex(qwen_quantize.PipelineError, "unsafe entry"):
                    qwen_quantize.cleanup_gguf_writer_temp(directory)
                self.assertTrue(directory.exists())

    def test_cleanup_rejects_unexpected_or_unsafe_temp_entries(self) -> None:
        cases = ("unexpected-name", "symlink", "directory", "hardlink", "mode")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw) / "private-tmp"
                directory.mkdir(mode=0o700)
                residue = directory / (
                    "foreign.bin" if case == "unexpected-name" else "tmpa1_b2c3d")
                if case == "symlink":
                    residue.symlink_to(Path(raw) / "outside")
                elif case == "directory":
                    residue.mkdir()
                else:
                    residue.write_bytes(b"spool")
                    residue.chmod(0o600)
                    if case == "hardlink":
                        os.link(residue, Path(raw) / "second-link")
                    elif case == "mode":
                        residue.chmod(0o644)
                with self.assertRaises(qwen_quantize.PipelineError):
                    qwen_quantize.cleanup_gguf_writer_temp(directory)
                self.assertTrue(directory.exists())

    def test_bounded_memory_execute_binds_cgroup_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            memory_max = fixture.root / "memory.max"
            swap_max = fixture.root / "memory.swap.max"
            memory_peak = fixture.root / "memory.peak"
            memory_max.write_text("2147483648\n", encoding="ascii")
            swap_max.write_text("0\n", encoding="ascii")
            memory_peak.write_text("536870912\n", encoding="ascii")
            command = fixture.command() + [
                "--bounded-memory-temp", "--gguf-splitter", str(fixture.gguf_splitter),
                "--conversion-memory-limit-bytes", "2147483648",
                "--cgroup-memory-max-path", str(memory_max),
                "--cgroup-memory-swap-max-path", str(swap_max),
                "--cgroup-memory-peak-path", str(memory_peak),
                "--execute",
            ]
            result = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(
                (fixture.work / "qwen-quant-build-record.json").read_text(encoding="utf-8")
            )
            measurement = record["conversion_memory"]["measurement"]
            self.assertEqual(measurement["status"], "measured_target_cgroup_v2")
            self.assertEqual(measurement["memory_limit_bytes"], 2147483648)
            self.assertEqual(measurement["swap_limit_bytes"], 0)
            self.assertEqual(measurement["cgroup_peak_after_conversion_bytes"], 536870912)
            self.assertEqual(
                record["conversion_memory"]["target_measurement_status"],
                "measured_within_pinned_no_swap_cgroup",
            )

    def test_bounded_memory_cgroup_contract_fails_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            memory_max = fixture.root / "memory.max"
            swap_max = fixture.root / "memory.swap.max"
            memory_peak = fixture.root / "memory.peak"
            memory_max.write_text("1073741824\n", encoding="ascii")
            swap_max.write_text("1\n", encoding="ascii")
            memory_peak.write_text("0\n", encoding="ascii")
            result = subprocess.run(
                fixture.command() + [
                    "--bounded-memory-temp", "--gguf-splitter", str(fixture.gguf_splitter),
                    "--conversion-memory-limit-bytes", "2147483648",
                    "--cgroup-memory-max-path", str(memory_max),
                    "--cgroup-memory-swap-max-path", str(swap_max),
                    "--cgroup-memory-peak-path", str(memory_peak),
                    "--execute",
                ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("memory.max differs", result.stderr)
            self.assertFalse(fixture.work.exists())

    def test_stock_control_verification_requires_exact_control_metadata(self) -> None:
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
                stock_control=True, quantized=True,
            )
            qwen_quantize.verify_gguf_set(
                [clean], expected, quantized=True, profile=profile,
                stock_control=True,
            )
            with self.assertRaisesRegex(qwen_quantize.PipelineError,
                                        "mutually exclusive"):
                qwen_quantize.verify_gguf_set(
                    [clean], expected, quantized=True, profile=profile,
                    intervention=TEST_INTERVENTION, stock_control=True,
                )
            edited = root / "edited.gguf"
            make_gguf(
                edited, split_no=0, split_count=1,
                tensor_name="per_layer_token_embd.weight", tensor_type=108,
                intervention=TEST_INTERVENTION, quantized=True,
            )
            with self.assertRaisesRegex(qwen_quantize.PipelineError,
                                        "pinned quantizer contract"):
                qwen_quantize.verify_gguf_set(
                    [edited], expected, quantized=True, profile=profile,
                    stock_control=True,
                )

    def test_output_audit_enforces_selected_mixed_tensor_types(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            profile = json.loads(
                (ROOT / "share" / "release_profiles" /
                 "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text()
            )
            expected_ple = {
                "ple_embedding.layer_multipliers": [1, 24_000_000_000_001],
                "ple_embedding.ngram_heads_offsets": [3, 5],
                "ple_embedding.ngram_heads_vocab_sizes": [7, 11],
            }
            q6_overrides = qwen_quantize.validated_quantization_arms(profile)[
                "rocmi4-q6k-embedding-head"]["per_tensor_overrides"]
            q6_tensors = [
                ("per_layer_token_embd.weight", 108),
                ("token_embd.weight", 14),
                ("output.weight", 14),
            ]
            q6_paths = []
            for index, (name, tensor_type) in enumerate(q6_tensors):
                path = root / f"q6-{index + 1:05d}-of-00003.gguf"
                make_gguf(path, split_no=index, split_count=3,
                          tensor_name=name, tensor_type=tensor_type,
                          stock_control=True, quantized=True)
                q6_paths.append(path)
            result = qwen_quantize.verify_gguf_set(
                q6_paths, expected_ple, quantized=True, profile=profile,
                stock_control=True, tensor_overrides=q6_overrides)
            self.assertEqual(result["tensor_type_counts"], {"14": 2, "108": 1})
            self.assertEqual(
                [item["matched_tensor_count"]
                 for item in result["tensor_override_evidence"]],
                [1, 1, 1],
            )
            make_gguf(q6_paths[-1], split_no=2, split_count=3,
                      tensor_name="output.weight", tensor_type=1,
                      stock_control=True, quantized=True)
            with self.assertRaisesRegex(qwen_quantize.PipelineError, "expected 14"):
                qwen_quantize.verify_gguf_set(
                    q6_paths, expected_ple, quantized=True, profile=profile,
                    stock_control=True, tensor_overrides=q6_overrides)

            fast_overrides = qwen_quantize.validated_quantization_arms(profile)[
                "rocmfp4-fast-matrix"]["per_tensor_overrides"]
            fast_tensors = [
                ("per_layer_token_embd.weight", 108),
                ("blk.0.attn_q.weight", 101),
                ("output_hc_down.weight", 101),
            ]
            fast_paths = []
            for index, (name, tensor_type) in enumerate(fast_tensors):
                path = root / f"fast-{index + 1:05d}-of-00003.gguf"
                make_gguf(path, split_no=index, split_count=3,
                          tensor_name=name, tensor_type=tensor_type,
                          stock_control=True, quantized=True)
                fast_paths.append(path)
            result = qwen_quantize.verify_gguf_set(
                fast_paths, expected_ple, quantized=True, profile=profile,
                stock_control=True, tensor_overrides=fast_overrides)
            self.assertEqual(result["tensor_type_counts"], {"101": 2, "108": 1})
            self.assertEqual(
                [item["matched_tensor_count"]
                 for item in result["tensor_override_evidence"]],
                [1, 1, 1],
            )

    def test_stock_control_executes_and_commits_clean_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            make_gguf(
                fixture.quant_split[0], split_no=0, split_count=2,
                tensor_name="per_layer_token_embd.weight", tensor_type=108,
                stock_control=True, quantized=True,
            )
            make_gguf(
                fixture.quant_split[1], split_no=1, split_count=2,
                tensor_name="blk.0.ffn_up.weight", tensor_type=108,
                stock_control=True, quantized=True,
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
            self.assertTrue(all(
                "Stock-Control" in Path(item["path"]).name
                and "Heretic" not in Path(item["path"]).name
                for item in record["output"]["shards"]
            ))

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

    def test_snapshot_allows_only_regular_root_fetch_lock(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            (fixture.snapshot / ".ember-fetch.lock").write_text(
                "fetch coordination\n", encoding="utf-8")
            result = subprocess.run(
                fixture.command(), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((fixture.root / "work.plan.json").read_text())
            self.assertEqual(
                record["snapshot"]["ignored_coordination_files"],
                [".ember-fetch.lock"],
            )

    def test_tool_provenance_fails_before_large_snapshot_integrity_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            (fixture.snapshot / "unexpected-source-file").write_text(
                "would fail the snapshot inventory\n", encoding="utf-8")
            (fixture.llama / "dirty-tool-file").write_text(
                "must fail first\n", encoding="utf-8")
            result = subprocess.run(
                fixture.command(), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("llama.cpp must differ from the pinned base only", result.stderr)
            self.assertNotIn("snapshot file inventory mismatch", result.stderr)

    def test_snapshot_rejects_fetch_lock_symlink_and_other_dotfile(self) -> None:
        for kind, expected in (("symlink", "regular non-symlink"),
                               ("other", "snapshot file inventory mismatch")):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                if kind == "symlink":
                    (fixture.snapshot / ".ember-fetch.lock").symlink_to("LICENSE")
                else:
                    (fixture.snapshot / ".another-fetch-file").write_text(
                        "not exempt\n", encoding="utf-8")
                result = subprocess.run(
                    fixture.command(), text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_snapshot_lease_rejects_active_fetcher_and_blocks_new_writer(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            lock_path = fixture.snapshot / ".ember-fetch.lock"
            with lock_path.open("w", encoding="utf-8") as writer:
                fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    fixture.command(), text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            self.assertEqual(result.returncode, 2)
            self.assertIn("download still owns the snapshot", result.stderr)
            with qwen_quantize.SnapshotReadLease(fixture.snapshot):
                with lock_path.open("w", encoding="utf-8") as writer:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_exact_companion_inventory_drives_combined_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            command = [*fixture.command(), *fixture.companion_args(enable_mmproj=True)]
            result = subprocess.run(
                [*command, "--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(
                (fixture.work / "qwen-quant-build-record.json").read_text())
            companion = record["companion_inventory"]
            self.assertEqual(companion["status"], "verified_exact")
            self.assertEqual(companion["enabled_roles"], ["mtp", "vision_mmproj"])
            self.assertEqual(companion["disabled_roles"], [])
            self.assertEqual(companion["fit_status"], "verified_exact_fit")
            self.assertFalse(companion["estimated_bytes_used"])
            self.assertTrue(all(row["regular_file"] and not row["symlink"]
                                for row in companion["roles"]))
            preflight = record["memory_preflight"]
            self.assertTrue(preflight["combined_fits"])
            self.assertEqual(
                preflight["combined_accounted_bytes"],
                preflight["artifact_bytes"] + preflight["runtime_reserve_bytes"]
                + companion["enabled_artifact_bytes"],
            )
            self.assertEqual(preflight["combined_fit_checks"], {
                "certification_memtotal": True,
                "device_budget": True,
                "live_gtt_cap": True,
            })
            self.assertEqual(
                companion["live_gtt_evidence"]["runner_gtt_cap_bytes"],
                133143986176,
            )
            self.assertEqual(
                record["experiment"]["eligibility_status"],
                "pending_canonical_live_gtt_evidence_and_measured_bakeoff_and_hardware_certification",
            )
            self.assertFalse(companion["live_gtt_evidence"]["authoritative_sysfs"])

    def test_main_quant_arm_and_mtp_matrix_contract_can_be_cross_paired(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            result = subprocess.run(
                [*fixture.command(), *fixture.companion_args(
                    mtp_matrix="Q4_0_ROCMFP4_FAST"),
                 "--mtp-matrix-quant-contract", "Q4_0_ROCMFP4_FAST"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((fixture.root / "work.plan.json").read_text())
            self.assertEqual(record["quantization_recipe"]["id"],
                             qwen_quantize.DEFAULT_QUANTIZATION_ARM)
            self.assertEqual(
                record["quantization_recipe"]["selected_mtp_matrix_quant_contract"],
                "Q4_0_ROCMFP4_FAST",
            )
            self.assertEqual(record["companion_inventory"]["roles"][0][
                "matrix_quant_contract"], "Q4_0_ROCMFP4_FAST")

    def test_exact_inventory_explicitly_disables_absent_mmproj(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            result = subprocess.run(
                [*fixture.command(), *fixture.companion_args()], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((fixture.root / "work.plan.json").read_text())
            companion = record["companion_inventory"]
            self.assertEqual(companion["enabled_roles"], ["mtp"])
            self.assertEqual(companion["disabled_roles"], ["vision_mmproj"])
            mmproj = companion["roles"][1]
            self.assertFalse(mmproj["enabled"])
            self.assertFalse(mmproj["artifact_present"])
            self.assertEqual(mmproj["counted_bytes"], 0)
            self.assertNotIn("path", mmproj)
            self.assertFalse(companion["release_companions_complete"])
            self.assertEqual(
                companion["pending_release_evidence"],
                ["canonical_live_gtt_evidence", "vision_mmproj"],
            )

    def test_companion_inventory_fails_closed_on_binding_format_and_file_type(self) -> None:
        cases = ("inventory_sha", "matrix", "symlink", "pages")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                fixture = Fixture(Path(raw))
                args = fixture.companion_args(
                    mtp_matrix=("Q4_0_ROCMFP4_FAST" if case == "matrix"
                                else "Q4_0_ROCMI4"))
                if case == "inventory_sha":
                    args[args.index("--companion-inventory-sha256") + 1] = "0" * 64
                elif case == "symlink":
                    inventory_path = Path(args[args.index("--companion-inventory") + 1])
                    inventory = json.loads(inventory_path.read_text())
                    mtp = Path(inventory["companions"][0]["path"])
                    target = fixture.root / "mtp-target.gguf"
                    mtp.rename(target)
                    mtp.symlink_to(target)
                elif case == "pages":
                    pages = Path(args[args.index("--ttm-pages-limit-path") + 1])
                    pages.write_text("32505857\n", encoding="utf-8")
                result = subprocess.run(
                    [*fixture.command(), *args], text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 2)
                self.assertTrue(any(fragment in result.stderr for fragment in (
                    "SHA-256 mismatch", "matrix quant contract",
                    "regular non-symlink", "live TTM pages_limit")), result.stderr)

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
                record["memory_preflight"]["certification_host_gtt_cap_bytes"],
                133143986176,
            )
            self.assertEqual(
                record["memory_preflight"]["companion_artifact_fit_status"],
                "pending_exact_companion_inventory",
            )
            self.assertIsNone(record["memory_preflight"]["combined_fits"])
            self.assertEqual(record["memory_preflight"]["shard_count"], 2)
            self.assertEqual(
                record["memory_preflight"]["shard_bytes"],
                [item["size_bytes"] for item in record["output"]["shards"]],
            )
            self.assertEqual(record["output"]["tensor_type_counts"], {"108": 2})
            self.assertEqual(
                record["output"]["tensor_override_evidence"], [{
                    "override": "^per_layer_token_embd\\.weight$=Q4_0_ROCMI4",
                    "format": "Q4_0_ROCMI4",
                    "ggml_tensor_type": 108,
                    "matched_tensor_count": 1,
                    "matched_tensor_names_sha256": hashlib.sha256(
                        b"per_layer_token_embd.weight").hexdigest(),
                }],
            )
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
            make_gguf(first, split_no=0, split_count=2,
                      tensor_name="per_layer_token_embd.weight", tensor_type=108,
                      intervention=TEST_INTERVENTION, quantized=True)
            make_gguf(second, split_no=1, split_count=2,
                      tensor_name="blk.0.ffn_up.weight", tensor_type=108,
                      intervention=TEST_INTERVENTION, quantized=True)
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

    def test_split_verification_accepts_canonical_quantized_continuation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            make_gguf(first, split_no=0, split_count=2,
                      tensor_name="per_layer_token_embd.weight", tensor_type=108,
                      intervention=TEST_INTERVENTION, quantized=True)
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
                      intervention=TEST_INTERVENTION, quantized=True)
            result = qwen_quantize.verify_gguf_set(
                [first, second], expected, quantized=True, profile=profile,
                intervention=TEST_INTERVENTION,
            )
            self.assertEqual(set(qwen_quantize.inspect_gguf(second)["metadata"]), {
                "split.no", "split.count", "split.tensors.count",
                "general.quantization_version", "general.file_type",
                "ember.intervention.kind",
                "ember.intervention.application_stage",
                "ember.intervention.manifest_sha256",
                "ember.intervention.target_names_sha256",
                "ember.intervention.target_count",
            })
            self.assertEqual(result["tensor_count"], 2)

    def test_split_verification_rejects_missing_or_unexpected_continuation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            make_gguf(first, split_no=0, split_count=2,
                      tensor_name="per_layer_token_embd.weight", tensor_type=108,
                      intervention=TEST_INTERVENTION, quantized=True)
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
                      intervention=TEST_INTERVENTION, quantized=True,
                      omit_metadata_keys={"split.count"})
            with self.assertRaisesRegex(qwen_quantize.PipelineError, "split.count is missing"):
                qwen_quantize.verify_gguf_set(
                    [first, second], expected, quantized=True, profile=profile,
                    intervention=TEST_INTERVENTION,
                )

            make_gguf(second, split_no=1, split_count=2,
                      tensor_name="blk.0.ffn_up.weight", tensor_type=108,
                      architecture="wrong", full_metadata=True,
                      intervention=TEST_INTERVENTION, quantized=True)
            with self.assertRaisesRegex(
                    qwen_quantize.PipelineError,
                    r"unexpected=.*general\.architecture"):
                qwen_quantize.verify_gguf_set(
                    [first, second], expected, quantized=True, profile=profile,
                    intervention=TEST_INTERVENTION,
                )

    def test_split_verification_rejects_tampered_quantizer_metadata(self) -> None:
        profile = json.loads(
            (ROOT / "share" / "release_profiles" /
             "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text()
        )
        expected = {
            "ple_embedding.layer_multipliers": [1, 24_000_000_000_001],
            "ple_embedding.ngram_heads_offsets": [3, 5],
            "ple_embedding.ngram_heads_vocab_sizes": [7, 11],
        }
        mutations = (
            ({"general.file_type"}, {}, "general.file_type"),
            (set(), {"general.quantization_version": (2, 2, None)},
             "general.quantization_version"),
            (set(), {"general.file_type": (4, 117, None)}, "general.file_type"),
            (set(), {"ember.injected": (8, "malicious", None)},
             r"unexpected=.*ember\.injected"),
        )
        for omitted, overrides, expected_error in mutations:
            with self.subTest(expected_error=expected_error), \
                    tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                first = root / "model-00001-of-00002.gguf"
                second = root / "model-00002-of-00002.gguf"
                make_gguf(
                    first, split_no=0, split_count=2,
                    tensor_name="per_layer_token_embd.weight", tensor_type=108,
                    intervention=TEST_INTERVENTION, quantized=True,
                )
                make_gguf(
                    second, split_no=1, split_count=2,
                    tensor_name="blk.0.ffn_up.weight", tensor_type=108,
                    intervention=TEST_INTERVENTION, quantized=True,
                    omit_metadata_keys=omitted,
                    metadata_overrides=overrides,
                )
                with self.assertRaisesRegex(qwen_quantize.PipelineError,
                                            expected_error):
                    qwen_quantize.verify_gguf_set(
                        [first, second], expected, quantized=True,
                        profile=profile, intervention=TEST_INTERVENTION,
                    )

    def test_execute_rolls_back_final_shards_after_semantic_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            first_metadata = qwen_quantize.inspect_gguf(
                fixture.quant_split[0])["metadata"]
            intervention = {
                key.removeprefix("ember.intervention."): field["value"]
                for key, field in first_metadata.items()
                if key.startswith("ember.intervention.")
            }
            make_gguf(
                fixture.quant_split[1], split_no=1, split_count=2,
                tensor_name="blk.0.ffn_up.weight", tensor_type=108,
                architecture="wrong", full_metadata=True,
                intervention=intervention, quantized=True,
            )
            result = subprocess.run(
                [*fixture.command(), "--execute"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("continuation shard 2 metadata", result.stderr)
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

    def test_memory_preflight_rejects_exact_companions_over_live_gtt_cap(self) -> None:
        profile = json.loads(
            (ROOT / "share" / "release_profiles" /
             "qwen3.8-flash-next-rocmi4-strix-halo.json").read_text()
        )
        gate = profile["quantization"]["native_262k_memory_gate"]
        companion_bytes = 4096
        artifact = (
            gate["certification_host_gtt_cap_bytes"]
            - gate["runtime_reserve_bytes"] - companion_bytes + 1
        )
        total = artifact + gate["runtime_reserve_bytes"]
        self.assertLess(total + companion_bytes,
                        gate["certification_host_memtotal_bytes"])
        companion = {
            "status": "verified_exact",
            "enabled_artifact_bytes": companion_bytes,
            "live_gtt_evidence": {
                "runner_gtt_cap_bytes": gate["certification_host_gtt_cap_bytes"],
            },
        }
        with self.assertRaisesRegex(qwen_quantize.PipelineError, "live_gtt_cap"):
            qwen_quantize.validate_memory_preflight({
                "artifact_bytes": artifact,
                "shard_count": 1,
                "shard_bytes": [artifact],
                "runtime_reserve_bytes": gate["runtime_reserve_bytes"],
                "budget_bytes": gate["device_budget_bytes"],
                "total_bytes": total,
                "headroom_bytes": gate["device_budget_bytes"] - total,
                "fits": True,
            }, profile, companion)

    def test_large_snapshot_digest_routes_through_direct_io_helper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "large-fixture.safetensors"
            path.write_bytes(b"large fixture bytes")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(qwen_quantize, "DIRECT_IO_MIN_BYTES", 1), \
                    mock.patch.object(
                        qwen_quantize, "snapshot_digest_direct",
                        return_value=expected,
                    ) as direct:
                actual, method = qwen_quantize.snapshot_artifact_digest(
                    path, path.stat().st_size, "sha256")
            self.assertEqual(actual, expected)
            self.assertEqual(method, "o_direct_dd_v1")
            direct.assert_called_once_with(path, path.stat().st_size, "sha256")

    def test_large_exact_file_hash_uses_identity_bound_direct_fd(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "large-companion.gguf"
            path.write_bytes(b"identity-bound companion")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(qwen_quantize, "DIRECT_IO_MIN_BYTES", 1), \
                    mock.patch.object(
                        qwen_quantize, "sha256_open_file",
                        wraps=qwen_quantize.sha256_open_file,
                    ) as bound_hash:
                evidence = qwen_quantize.inspect_exact_file(
                    path.resolve(), expected, path.stat().st_size, "companion")
            self.assertEqual(evidence["integrity_read_method"], "o_direct_bound_fd_v1")
            self.assertEqual(bound_hash.call_args.args[1], path.stat().st_size)
            self.assertEqual(bound_hash.call_args.args[2], path.resolve())

    def test_direct_git_blob_digest_hashes_exact_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "blob"
            path.write_bytes(b"git blob payload")
            fake_dd = mock.Mock()
            fake_dd.stdout = mock.Mock()
            fake_dd.stdout.read = mock.Mock(side_effect=[path.read_bytes(), b""])
            fake_dd.stderr = mock.Mock()
            fake_dd.stderr.read.return_value = b""
            fake_dd.wait.return_value = 0
            with mock.patch.object(qwen_quantize.shutil, "which", return_value="/usr/bin/dd"), \
                    mock.patch.object(
                        qwen_quantize.subprocess, "Popen", return_value=fake_dd,
                    ) as popen:
                actual = qwen_quantize.snapshot_digest_direct(
                    path, path.stat().st_size, "git_blob_sha1")
            self.assertEqual(actual, git_blob(path))
            self.assertIn("iflag=direct", popen.call_args.args[0])

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
