#!/usr/bin/env python3
"""Extract Qwen3.8 per-layer directions and write a quantizer manifest.

This is an offline, non-publishing stage.  It deliberately captures every
decoder layer's 2560-wide residual-writer output: ``linear_attn.out_proj`` on
GDN layers and ``self_attn.o_proj`` on QSA layers.  Qwen4Exp's ordinary decoder
hidden states are 10240-wide hyper-connection state and are not valid
left-projection directions for these 2560-row output matrices.

The primary backend consumes the raw F32 activation dumps produced by Ember's
Qwen runtime, matching the pinned OtherU architecture-change workflow.  Each
record is exactly 48*2560 little-endian floats in numeric layer order.  A stock
ROCMI4 control model can produce these dumps on the 128 GiB target; loading the
~335 GiB BF16 checkpoint is not required.  A Transformers hook backend remains
available as an optional reference path on a sufficiently large/offloaded host.

Input corpora are bounded JSONL files.  Every row must contain a unique ``id``
and text-only chat ``messages``.  Good, bad, and held-out evaluation content is
canonicalized and checked for pairwise overlap before the model is loaded.
Only batch-sized prompt data and one float64 sum per layer are retained.

The resulting JSON is the exact ``--intervention-manifest`` input accepted by
``scripts/qwen_quantize.py`` and ``ember-gguf-quantize``.  This program does not
measure efficacy, choose a projection strength, quantize, or publish.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Callable, Iterable, Iterator, Sequence

import qwen_quantize


class InterventionError(ValueError):
    pass


@dataclass(frozen=True)
class ArchitectureSpec:
    layer_count: int
    hidden_size: int
    writer_input_size: int
    qsa_layers: frozenset[int]


QWEN_SPEC = ArchitectureSpec(
    layer_count=48,
    hidden_size=2560,
    writer_input_size=6144,
    qsa_layers=frozenset(range(3, 48, 4)),
)

QWEN_POLICY_LAYERS = {
    "band-10-42": frozenset(range(10, 43)),
    "upper-24": frozenset(range(24, 48)),
    "upper-12": frozenset(range(36, 48)),
    "non-qsa-band-10-42": frozenset(
        layer for layer in range(10, 43) if layer not in QWEN_SPEC.qsa_layers
    ),
}
QWEN_POLICY_EVIDENCE = {
    "source_revision": "a3c6a728510f91394e991504951ac316cd3a89af",
    "deepseek_reference_band": "10-42",
    "qwen_status": "exploratory_transfer_hypothesis",
}


@dataclass(frozen=True)
class CorpusRecord:
    record_id: str
    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class CorpusEvidence:
    path: Path
    corpus_id: str
    sha256: str
    record_count: int
    content_fingerprints: frozenset[str]


class HookHandle:
    """Structural type documentation for torch and test hook handles."""

    def remove(self) -> None:  # pragma: no cover - protocol-shaped only
        raise NotImplementedError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_messages(messages: Sequence[dict[str, str]]) -> bytes:
    return json.dumps(
        {"messages": list(messages)}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def parse_corpus_record(value: Any, label: str) -> CorpusRecord:
    if not isinstance(value, dict):
        raise InterventionError(f"{label} must be a JSON object")
    record_id = value.get("id")
    messages = value.get("messages")
    if not isinstance(record_id, str) or not record_id or len(record_id) > 256:
        raise InterventionError(f"{label}.id must be a nonempty string of at most 256 characters")
    if not isinstance(messages, list) or not messages:
        raise InterventionError(f"{label}.messages must be a nonempty array")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise InterventionError(f"{label}.messages[{index}] must be an object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise InterventionError(
                f"{label}.messages[{index}] must contain string role/content fields"
            )
        # Direction extraction is text-only.  Accepting multimodal lists here
        # would silently make processor and image preprocessing part of the
        # extraction corpus contract.
        normalized.append({"role": role, "content": content})
    return CorpusRecord(record_id=record_id, messages=tuple(normalized))


def iter_jsonl(
    path: Path, *, max_records: int, max_line_bytes: int,
) -> Iterator[CorpusRecord]:
    if not path.is_file():
        raise InterventionError(f"corpus does not exist: {path}")
    count = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(max_line_bytes + 1)
            if not raw:
                break
            if len(raw) > max_line_bytes:
                raise InterventionError(
                    f"corpus line exceeds --max-line-bytes in {path}"
                )
            if not raw.strip():
                continue
            count += 1
            if count > max_records:
                raise InterventionError(f"corpus exceeds --max-records: {path}")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InterventionError(f"invalid JSONL row {count} in {path}: {exc}") from exc
            yield parse_corpus_record(value, f"{path}:{count}")
    if count == 0:
        raise InterventionError(f"corpus contains no records: {path}")


def scan_corpus(
    path: Path, corpus_id: str, *, max_records: int, max_line_bytes: int,
) -> CorpusEvidence:
    ids: set[str] = set()
    fingerprints: set[str] = set()
    count = 0
    for record in iter_jsonl(path, max_records=max_records, max_line_bytes=max_line_bytes):
        if record.record_id in ids:
            raise InterventionError(f"duplicate corpus id {record.record_id!r} in {path}")
        ids.add(record.record_id)
        fingerprint = hashlib.sha256(canonical_messages(record.messages)).hexdigest()
        if fingerprint in fingerprints:
            raise InterventionError(f"duplicate canonical prompt content in {path}")
        fingerprints.add(fingerprint)
        count += 1
    return CorpusEvidence(
        path=path,
        corpus_id=corpus_id,
        sha256=sha256_file(path),
        record_count=count,
        content_fingerprints=frozenset(fingerprints),
    )


def require_disjoint_corpora(
    good: CorpusEvidence, bad: CorpusEvidence, held_out: CorpusEvidence,
) -> None:
    pairs = (
        (good, bad, "good/bad extraction"),
        (good, held_out, "good/held-out"),
        (bad, held_out, "bad/held-out"),
    )
    for left, right, label in pairs:
        overlap = left.content_fingerprints & right.content_fingerprints
        if overlap:
            raise InterventionError(
                f"{label} corpora overlap by {len(overlap)} canonical prompt(s)"
            )


def activation_dump_means(
    path: Path,
    *,
    expected_records: int,
    spec: ArchitectureSpec,
    winsorization_quantile: float,
) -> tuple[list[list[float]], str]:
    """Stream one pinned-OtherU raw F32 dump into per-layer float64 means."""
    if sys.byteorder != "little":
        raise InterventionError("Qwen activation dumps require a little-endian host")
    if not path.is_file() or path.is_symlink():
        raise InterventionError(f"activation dump must be a regular non-symlink file: {path}")
    values_per_record = spec.layer_count * spec.hidden_size
    bytes_per_record = values_per_record * 4
    expected_bytes = expected_records * bytes_per_record
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise InterventionError(
            f"activation dump {path} has {actual_bytes} bytes; expected exactly "
            f"{expected_bytes} for {expected_records} records"
        )
    sums = [[0.0] * spec.hidden_size for _ in range(spec.layer_count)]
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        for record_index in range(expected_records):
            raw = stream.read(bytes_per_record)
            if len(raw) != bytes_per_record:
                raise InterventionError(
                    f"short activation record {record_index} in {path}"
                )
            digest.update(raw)
            values = struct.unpack(f"<{values_per_record}f", raw)
            for layer in range(spec.layer_count):
                start = layer * spec.hidden_size
                row = values[start:start + spec.hidden_size]
                if any(not math.isfinite(value) for value in row):
                    raise InterventionError(
                        f"non-finite activation in {path}, record {record_index}, layer {layer}"
                    )
                threshold = math.inf
                if winsorization_quantile < 1.0:
                    ordered = sorted(abs(float(value)) for value in row)
                    position = winsorization_quantile * (len(ordered) - 1)
                    lower = int(math.floor(position))
                    upper = int(math.ceil(position))
                    fraction = position - lower
                    threshold = (
                        ordered[lower] * (1.0 - fraction)
                        + ordered[upper] * fraction
                    )
                layer_sum = sums[layer]
                for column, value in enumerate(row):
                    clipped = max(-threshold, min(threshold, float(value)))
                    layer_sum[column] += clipped
        if stream.read(1):
            raise InterventionError(f"activation dump grew while being read: {path}")
    means = [
        [value / expected_records for value in layer_sum]
        for layer_sum in sums
    ]
    return means, digest.hexdigest()


def locate_qwen_layers(model: Any, spec: ArchitectureSpec) -> Sequence[Any]:
    candidates: list[Any] = []
    direct = getattr(model, "model", None)
    if direct is not None:
        language_model = getattr(direct, "language_model", None)
        if language_model is not None:
            candidates.append(getattr(language_model, "layers", None))
        candidates.append(getattr(direct, "layers", None))
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        candidates.append(getattr(language_model, "layers", None))
    for layers in candidates:
        if layers is not None:
            try:
                count = len(layers)
            except TypeError:
                continue
            if count != spec.layer_count:
                raise InterventionError(
                    f"Qwen decoder has {count} layers, expected {spec.layer_count}"
                )
            return layers
    raise InterventionError("cannot locate Qwen language-model decoder layers")


def residual_writer(layer: Any, layer_index: int, spec: ArchitectureSpec) -> Any:
    if layer_index in spec.qsa_layers:
        component = getattr(layer, "self_attn", None)
        writer = getattr(component, "o_proj", None)
        label = "self_attn.o_proj"
    else:
        component = getattr(layer, "linear_attn", None)
        writer = getattr(component, "out_proj", None)
        label = "linear_attn.out_proj"
    if writer is None:
        raise InterventionError(f"Qwen decoder layer {layer_index} lacks {label}")
    return writer


class PythonTensorOps:
    """Small dependency-free tensor adapter used by deterministic tests."""

    @staticmethod
    def last_token_batch_sum(writer_output: Any, dimension: int, quantile: float) -> list[float]:
        if not isinstance(writer_output, list) or not writer_output:
            raise InterventionError("mock writer output must be a nonempty batch list")
        result = [0.0] * dimension
        for sample in writer_output:
            if not isinstance(sample, list) or not sample:
                raise InterventionError("mock writer output sample must contain positions")
            row = [float(value) for value in sample[-1]]
            if len(row) != dimension or any(not math.isfinite(value) for value in row):
                raise InterventionError("mock writer output has the wrong final dimension")
            if quantile < 1.0:
                ordered = sorted(abs(value) for value in row)
                position = quantile * (len(ordered) - 1)
                lower = int(math.floor(position))
                upper = int(math.ceil(position))
                fraction = position - lower
                threshold = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
                row = [max(-threshold, min(threshold, value)) for value in row]
            for index, value in enumerate(row):
                result[index] += value
        return result


class TorchTensorOps:
    def __init__(self, torch_module: Any):
        self.torch = torch_module

    def last_token_batch_sum(self, writer_output: Any, dimension: int, quantile: float) -> list[float]:
        if getattr(writer_output, "ndim", None) != 3 or writer_output.shape[-1] != dimension:
            raise InterventionError(
                f"residual-writer output must have final dimension {dimension}"
            )
        # Left padding makes -1 the prompt frontier. Hooking the output
        # projection itself keeps this vector in the writer's 2560-row space.
        frontier = writer_output[:, -1, :].detach().to(dtype=self.torch.float32)
        if quantile < 1.0:
            threshold = self.torch.quantile(
                frontier.abs(), quantile, dim=1, keepdim=True
            )
            frontier = self.torch.clamp(frontier, min=-threshold, max=threshold)
        values = frontier.to(device="cpu", dtype=self.torch.float64).sum(dim=0)
        if not bool(self.torch.isfinite(values).all()):
            raise InterventionError("non-finite Qwen residual-writer activation")
        return [float(value) for value in values.tolist()]


def model_input_device(model: Any) -> Any | None:
    getter = getattr(model, "get_input_embeddings", None)
    if not callable(getter):
        return None
    embeddings = getter()
    return getattr(getattr(embeddings, "weight", None), "device", None)


def move_batch(batch: dict[str, Any], device: Any | None) -> dict[str, Any]:
    if device is None:
        return batch
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        to = getattr(value, "to", None)
        moved[key] = to(device) if callable(to) else value
    return moved


def accumulate_activation_means(
    model: Any,
    batches: Iterable[tuple[dict[str, Any], int]],
    *,
    spec: ArchitectureSpec,
    tensor_ops: Any,
    winsorization_quantile: float,
    inference_context: Callable[[], Any] = nullcontext,
) -> tuple[list[list[float]], int]:
    """Stream batches through Qwen and return one mean writer output per layer."""
    if not 0.0 <= winsorization_quantile <= 1.0:
        raise InterventionError("winsorization quantile must be in [0, 1]")
    layers = locate_qwen_layers(model, spec)
    sums = [[0.0] * spec.hidden_size for _ in range(spec.layer_count)]
    calls = [0] * spec.layer_count
    handles: list[Any] = []

    def make_hook(layer_index: int) -> Callable[..., None]:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            batch_sum = tensor_ops.last_token_batch_sum(
                output, spec.hidden_size, winsorization_quantile
            )
            if len(batch_sum) != spec.hidden_size:
                raise InterventionError(f"layer {layer_index} activation sum has wrong width")
            for column, value in enumerate(batch_sum):
                sums[layer_index][column] += float(value)
            calls[layer_index] += 1

        return hook

    for index, layer in enumerate(layers):
        writer = residual_writer(layer, index, spec)
        register = getattr(writer, "register_forward_hook", None)
        if not callable(register):
            raise InterventionError(
                f"layer {index} residual writer cannot register a hook"
            )
        handles.append(register(make_hook(index)))

    total = 0
    input_device = model_input_device(model)
    try:
        with inference_context():
            for batch, record_count in batches:
                if record_count < 1:
                    raise InterventionError("encoded batch record count must be positive")
                before = list(calls)
                forwarded = move_batch(batch, input_device)
                # output_hidden_states is intentionally absent: those states are
                # 10240-wide on Qwen4Exp. The hybrid writer hooks above preserve
                # Heretic's post-writer output-space contract at width 2560.
                model(**forwarded, use_cache=False, return_dict=True)
                for index in range(spec.layer_count):
                    if calls[index] != before[index] + 1:
                        raise InterventionError(
                            f"layer {index} residual-writer hook fired {calls[index] - before[index]} times"
                        )
                total += record_count
    finally:
        for handle in handles:
            handle.remove()
    if total == 0:
        raise InterventionError("activation extraction processed no records")
    means = [[value / total for value in layer_sum] for layer_sum in sums]
    return means, total


def rendered_batches(
    path: Path,
    tokenizer: Any,
    *,
    batch_size: int,
    max_input_tokens: int,
    max_records: int,
    max_line_bytes: int,
) -> Iterator[tuple[dict[str, Any], int]]:
    pending: list[list[int]] = []
    tokenizer.padding_side = "left"
    for record in iter_jsonl(path, max_records=max_records, max_line_bytes=max_line_bytes):
        rendered = tokenizer.apply_chat_template(
            list(record.messages), tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_attention_mask=False
        )
        token_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        if not isinstance(token_ids, list) or any(not isinstance(token, int) for token in token_ids):
            raise InterventionError("tokenizer did not return a flat input_ids list")
        if not token_ids:
            raise InterventionError(f"tokenizer produced an empty prompt for record {record.record_id}")
        if len(token_ids) > max_input_tokens:
            raise InterventionError(
                f"record {record.record_id} has {len(token_ids)} tokens, exceeds --max-input-tokens"
            )
        pending.append(token_ids)
        if len(pending) == batch_size:
            yield tokenizer.pad({"input_ids": pending}, padding=True, return_tensors="pt"), len(pending)
            pending = []
    if pending:
        yield tokenizer.pad({"input_ids": pending}, padding=True, return_tensors="pt"), len(pending)


def normalize(values: Sequence[float], label: str) -> list[float]:
    norm = math.sqrt(math.fsum(float(value) * float(value) for value in values))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise InterventionError(f"{label} is degenerate")
    return [float(value) / norm for value in values]


def as_f32_unit(values: Sequence[float], label: str) -> list[float]:
    normalized = normalize(values, label)
    for _ in range(2):
        rounded = [struct.unpack("<f", struct.pack("<f", value))[0] for value in normalized]
        normalized = normalize(rounded, label)
    return [struct.unpack("<f", struct.pack("<f", value))[0] for value in normalized]


def build_directions(
    good_means: Sequence[Sequence[float]],
    bad_means: Sequence[Sequence[float]],
    *,
    orthogonalize_control_mean: bool,
    spec: ArchitectureSpec,
) -> list[list[float]]:
    if len(good_means) != spec.layer_count or len(bad_means) != spec.layer_count:
        raise InterventionError("activation means do not cover every Qwen layer")
    directions: list[list[float]] = []
    for layer in range(spec.layer_count):
        good = [float(value) for value in good_means[layer]]
        bad = [float(value) for value in bad_means[layer]]
        if len(good) != spec.hidden_size or len(bad) != spec.hidden_size:
            raise InterventionError(f"layer {layer} activation mean has the wrong width")
        direction = [bad[index] - good[index] for index in range(spec.hidden_size)]
        if orthogonalize_control_mean:
            good_unit = normalize(good, f"layer {layer} good activation mean")
            direction_unit = normalize(direction, f"layer {layer} bad-good direction")
            projection = math.fsum(
                direction_unit[index] * good_unit[index]
                for index in range(spec.hidden_size)
            )
            direction = [
                direction_unit[index] - projection * good_unit[index]
                for index in range(spec.hidden_size)
            ]
        directions.append(as_f32_unit(direction, f"layer {layer} direction"))
    return directions


def packed_f32_sha256(values: Sequence[float]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<f", value) for value in values)).hexdigest()


def writer_name(layer: int, spec: ArchitectureSpec) -> str:
    suffix = "attn_output" if layer in spec.qsa_layers else "ssm_out"
    return f"blk.{layer}.{suffix}.weight"


def read_layer_scales(args: argparse.Namespace, spec: ArchitectureSpec) -> list[float]:
    if args.scale is not None:
        # Pinned OtherU revision a3c6a728 found 10..42 coherence-preserving
        # for DeepSeek and found that editing 0..9 broke agentic coherence.
        # Treat that band as an exploratory transfer hypothesis for Qwen; the
        # held-out Qwen gates remain authoritative.
        values = [
            float(args.scale) if 10 <= layer <= 42 else 0.0
            for layer in range(spec.layer_count)
        ]
    else:
        try:
            raw = json.loads(args.layer_scales.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InterventionError(f"cannot read --layer-scales: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != {str(index) for index in range(spec.layer_count)}:
            raise InterventionError(
                f"--layer-scales must map every layer 0..{spec.layer_count - 1} exactly once"
            )
        values = []
        for index in range(spec.layer_count):
            value = raw[str(index)]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InterventionError(f"layer scale {index} must be numeric")
            values.append(float(value))
    if any(not math.isfinite(value) or abs(value) > 16.0 for value in values):
        raise InterventionError("layer scales must be finite and within [-16, 16]")
    if not any(value != 0.0 for value in values):
        raise InterventionError("at least one layer scale must be non-zero")
    return values


def validate_layer_policy(
    layer_policy: str, layer_scales: Sequence[float], spec: ArchitectureSpec,
) -> None:
    if spec != QWEN_SPEC:
        return
    expected = QWEN_POLICY_LAYERS.get(layer_policy)
    if expected is None:
        raise InterventionError(f"unknown Qwen layer policy: {layer_policy}")
    actual = frozenset(
        layer for layer, scale in enumerate(layer_scales) if float(scale) != 0.0
    )
    if actual != expected:
        raise InterventionError(
            f"Qwen layer policy {layer_policy} does not match its exact target layers"
        )


def infer_layer_policy(layer_scales: Sequence[float], spec: ArchitectureSpec) -> str:
    if spec != QWEN_SPEC:
        return "explicit-layer-scales"
    actual = frozenset(
        layer for layer, scale in enumerate(layer_scales) if float(scale) != 0.0
    )
    matches = [name for name, expected in QWEN_POLICY_LAYERS.items()
               if actual == expected]
    if len(matches) != 1:
        raise InterventionError(
            "Qwen layer scales must match one exact exploratory bakeoff policy"
        )
    return matches[0]


def build_manifest(
    *,
    profile: dict[str, Any],
    directions: Sequence[Sequence[float]],
    layer_scales: Sequence[float],
    good: CorpusEvidence,
    bad: CorpusEvidence,
    held_out: CorpusEvidence,
    good_count: int,
    bad_count: int,
    orthogonalize_control_mean: bool,
    winsorization_quantile: float,
    max_input_tokens: int,
    batch_size: int,
    load_mode: str,
    layer_policy: str,
    activation_evidence: dict[str, Any] | None = None,
    spec: ArchitectureSpec = QWEN_SPEC,
) -> dict[str, Any]:
    if len(directions) != spec.layer_count or len(layer_scales) != spec.layer_count:
        raise InterventionError("manifest inputs do not cover every Qwen layer")
    validate_layer_policy(layer_policy, layer_scales, spec)
    direction_rows = []
    for layer, values in enumerate(directions):
        if len(values) != spec.hidden_size:
            raise InterventionError(f"direction {layer} has wrong width")
        direction_rows.append({
            "id": f"layer-{layer:02d}-residual-writer-output-r1",
            "dtype": "F32",
            "values": list(values),
            "sha256": packed_f32_sha256(values),
            "layer": layer,
            "activation": "residual_writer.output",
        })
    targets = []
    for layer, scale in enumerate(layer_scales):
        if scale == 0.0:
            continue
        targets.append({
            "tensor_name": writer_name(layer, spec),
            "direction_id": f"layer-{layer:02d}-residual-writer-output-r1",
            "scale": scale,
            "normalization": "row_norm_preserve",
            "expected_shape": [spec.writer_input_size, spec.hidden_size],
        })
    targets.sort(key=lambda target: target["tensor_name"])
    target_names = [target["tensor_name"] for target in targets]
    target_names_sha = hashlib.sha256("\n".join(target_names).encode("utf-8")).hexdigest()
    source = profile["source"]
    contract = profile["intervention"]
    return {
        "schema_version": 1,
        "kind": "directional_ablation",
        "status": "complete",
        "weight_intervention": True,
        "prompt_only": False,
        "application_stage": "pre_quantization_encoding",
        "source": {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "snapshot_inventory_sha256": source["snapshot_inventory_sha256"],
        },
        "tooling": {
            "otheru_quant_pipeline": contract["otheru_pipeline"],
            "upstream_heretic": contract["upstream_heretic"],
            "extractor": {
                "implementation": "ember-qwen-residual-writer-activation-extractor",
                "schema_version": 2,
            },
        },
        "corpora": [
            {
                "id": good.corpus_id,
                "class": "good_control",
                "role": "direction_extraction",
                "sha256": good.sha256,
                "record_count": good.record_count,
                "held_out_evaluation_overlap_count": 0,
            },
            {
                "id": bad.corpus_id,
                "class": "bad_target",
                "role": "direction_extraction",
                "sha256": bad.sha256,
                "record_count": bad.record_count,
                "held_out_evaluation_overlap_count": 0,
            },
        ],
        "held_out_evaluation": {
            "id": held_out.corpus_id,
            "sha256": held_out.sha256,
            "record_count": held_out.record_count,
            "overlap_count": 0,
            "comparison": "canonical_text_chat_messages_sha256",
        },
        "extraction": {
            "direction_scope": "per_layer",
            "layer_count": spec.layer_count,
            "activation_width": spec.hidden_size,
            "semantic_capture_point": "decoder_layer.residual_writer.output",
            "transformers_hook_module": (
                "model.language_model.layers.N.{linear_attn.out_proj|self_attn.o_proj}"
            ),
            "transformers_hook_value": "forward_output[:,-1,:]",
            "layer_policy": layer_policy,
            "policy_evidence": dict(QWEN_POLICY_EVIDENCE),
            "hidden_states_api_used": False,
            "good_records_processed": good_count,
            "bad_records_processed": bad_count,
            "orthogonalize_control_mean": orthogonalize_control_mean,
            "winsorization_quantile": winsorization_quantile,
            "max_input_tokens": max_input_tokens,
            "batch_size": batch_size,
            "load_mode": load_mode,
            "streaming": True,
            "efficacy_evaluated": False,
            "activation_evidence": activation_evidence or {
                "backend": "transformers_optional_reference",
            },
        },
        "directions": direction_rows,
        "targets": targets,
        "tensor_map": {
            "kind": "exact_tensor_names",
            "target_count": len(targets),
            "target_names_sha256": target_names_sha,
            "candidate_writer_count": spec.layer_count,
            "mtp_omitted": True,
            "vision_omitted": True,
        },
    }


def write_manifest_noreplace(
    output: Path, manifest: dict[str, Any], profile: dict[str, Any]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise InterventionError(f"refusing to overwrite intervention manifest: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Validate with the exact consumer before making the manifest visible.
        qwen_quantize.validate_intervention_manifest(temporary, profile)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise InterventionError(f"refusing to overwrite intervention manifest: {output}") from exc
        qwen_quantize.fsync_directory(output.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_max_memory(entries: Sequence[str]) -> dict[int | str, str] | None:
    if not entries:
        return None
    result: dict[int | str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise InterventionError("--max-memory entries must be DEVICE=SIZE")
        device, size = entry.split("=", 1)
        key: int | str = int(device) if device.isdigit() else device
        if key in result or not size:
            raise InterventionError(f"invalid or duplicate --max-memory entry: {entry}")
        result[key] = size
    return result


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise InterventionError(
            "real extraction requires torch, transformers, accelerate, and bitsandbytes for bnb-4bit"
        ) from exc
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
        "device_map": args.device_map,
        "offload_state_dict": True,
    }
    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if args.offload_dir is not None:
        args.offload_dir.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(args.offload_dir)
    if args.load_mode == "bnb-4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["dtype"] = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(args.snapshot_dir, **kwargs)
    model.eval()
    return model, tokenizer, torch


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path,
        default=root / "share/release_profiles/qwen3.8-flash-next-rocmi4-strix-halo.json",
    )
    parser.add_argument(
        "--activation-backend", choices=("dump", "transformers"), default="dump",
        help="raw Ember/OtherU-compatible F32 dumps are the 128 GiB-safe default",
    )
    parser.add_argument("--snapshot-dir", type=Path,
                        help="required only by the optional Transformers backend")
    parser.add_argument("--snapshot-revision",
                        help="required only by the optional Transformers backend")
    parser.add_argument("--good-corpus", type=Path, required=True)
    parser.add_argument("--bad-corpus", type=Path, required=True)
    parser.add_argument("--held-out-corpus", type=Path, required=True)
    parser.add_argument("--good-activations", type=Path,
                        help="raw 48x2560 F32 records in good-corpus order")
    parser.add_argument("--bad-activations", type=Path,
                        help="raw 48x2560 F32 records in bad-corpus order")
    parser.add_argument(
        "--activation-artifact-sha256",
        help="supplied SHA-256 evidence for the stock ROCMI4 control GGUF; the artifact is not rehashed here",
    )
    parser.add_argument("--good-corpus-id", default="qwen-good-control-v1")
    parser.add_argument("--bad-corpus-id", default="qwen-bad-target-v1")
    parser.add_argument("--held-out-corpus-id", default="qwen-held-out-eval-v1")
    parser.add_argument("--output", type=Path, required=True)
    scales = parser.add_mutually_exclusive_group(required=True)
    scales.add_argument(
        "--scale", type=float,
        help="projection strength for the exploratory layers 10..42 band",
    )
    scales.add_argument(
        "--layer-scales", type=Path,
        help="JSON object mapping every layer 0..47 to an explicit scale; zero omits a target",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-input-tokens", type=int, default=0,
        help="required and positive only for the optional Transformers backend",
    )
    parser.add_argument("--max-records", type=int, default=100000)
    parser.add_argument("--max-line-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--winsorization-quantile", type=float, default=1.0)
    parser.add_argument(
        "--no-orthogonalize-control-mean", action="store_true",
        help="disable Heretic's default projection of each direction off its good mean",
    )
    parser.add_argument("--load-mode", choices=("bnb-4bit", "bf16"), default="bnb-4bit")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--max-memory", action="append", default=[], metavar="DEVICE=SIZE",
        help="repeatable Transformers device-map limit, e.g. 0=120GiB or cpu=200GiB",
    )
    parser.add_argument("--offload-dir", type=Path)
    return parser.parse_args(list(argv))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.max_input_tokens < 0:
        raise InterventionError("--batch-size must be positive and --max-input-tokens non-negative")
    if args.max_records < 1 or args.max_line_bytes < 128:
        raise InterventionError("corpus bounds must be positive and --max-line-bytes at least 128")
    if not 0.0 <= args.winsorization_quantile <= 1.0:
        raise InterventionError("--winsorization-quantile must be in [0, 1]")
    profile, inventory, _inventory_path = qwen_quantize.validate_profile(args.profile.resolve())
    good = scan_corpus(
        args.good_corpus.resolve(), args.good_corpus_id,
        max_records=args.max_records, max_line_bytes=args.max_line_bytes,
    )
    bad = scan_corpus(
        args.bad_corpus.resolve(), args.bad_corpus_id,
        max_records=args.max_records, max_line_bytes=args.max_line_bytes,
    )
    held_out = scan_corpus(
        args.held_out_corpus.resolve(), args.held_out_corpus_id,
        max_records=args.max_records, max_line_bytes=args.max_line_bytes,
    )
    require_disjoint_corpora(good, bad, held_out)
    layer_scales = read_layer_scales(args, QWEN_SPEC)
    if args.activation_backend == "dump":
        if args.good_activations is None or args.bad_activations is None:
            raise InterventionError(
                "dump backend requires --good-activations and --bad-activations"
            )
        artifact_sha = args.activation_artifact_sha256
        if not isinstance(artifact_sha, str) or qwen_quantize.SHA256_RE.fullmatch(artifact_sha) is None:
            raise InterventionError(
                "dump backend requires supplied --activation-artifact-sha256 evidence for the stock ROCMI4 control"
            )
        good_means, good_dump_sha = activation_dump_means(
            args.good_activations.resolve(), expected_records=good.record_count,
            spec=QWEN_SPEC, winsorization_quantile=args.winsorization_quantile,
        )
        bad_means, bad_dump_sha = activation_dump_means(
            args.bad_activations.resolve(), expected_records=bad.record_count,
            spec=QWEN_SPEC, winsorization_quantile=args.winsorization_quantile,
        )
        good_count, bad_count = good.record_count, bad.record_count
        activation_evidence = {
            "backend": "ember_qwen_runtime_f32_dump",
            "format": "48x2560-little-endian-f32-writer-output-records-v2",
            "record_order": "corpus_jsonl_order",
            "stock_rocmi4_artifact_sha256": artifact_sha,
            "artifact_sha256_verification": "supplied_not_locally_rehashed",
            "good_dump_sha256": good_dump_sha,
            "bad_dump_sha256": bad_dump_sha,
            "good_dump_bytes": args.good_activations.stat().st_size,
            "bad_dump_bytes": args.bad_activations.stat().st_size,
        }
        load_mode = "stock_rocmi4_runtime_f32_dump"
    else:
        if args.snapshot_dir is None or args.snapshot_revision is None:
            raise InterventionError(
                "Transformers backend requires --snapshot-dir and --snapshot-revision"
            )
        if args.max_input_tokens < 1:
            raise InterventionError(
                "Transformers backend requires a positive --max-input-tokens"
            )
        qwen_quantize.validate_snapshot(
            args.snapshot_dir.resolve(), args.snapshot_revision, profile, inventory
        )
        model, tokenizer, torch = load_model_and_tokenizer(args)
        tensor_ops = TorchTensorOps(torch)
        good_means, good_count = accumulate_activation_means(
            model,
            rendered_batches(
                good.path, tokenizer, batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens, max_records=args.max_records,
                max_line_bytes=args.max_line_bytes,
            ),
            spec=QWEN_SPEC,
            tensor_ops=tensor_ops,
            winsorization_quantile=args.winsorization_quantile,
            inference_context=torch.inference_mode,
        )
        bad_means, bad_count = accumulate_activation_means(
            model,
            rendered_batches(
                bad.path, tokenizer, batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens, max_records=args.max_records,
                max_line_bytes=args.max_line_bytes,
            ),
            spec=QWEN_SPEC,
            tensor_ops=tensor_ops,
            winsorization_quantile=args.winsorization_quantile,
            inference_context=torch.inference_mode,
        )
        activation_evidence = {
            "backend": "transformers_optional_reference",
            "snapshot_revision": args.snapshot_revision,
        }
        load_mode = args.load_mode
    if good_count != good.record_count or bad_count != bad.record_count:
        raise InterventionError("activation extraction count differs from scanned corpus evidence")
    directions = build_directions(
        good_means, bad_means,
        orthogonalize_control_mean=not args.no_orthogonalize_control_mean,
        spec=QWEN_SPEC,
    )
    manifest = build_manifest(
        profile=profile,
        directions=directions,
        layer_scales=layer_scales,
        good=good,
        bad=bad,
        held_out=held_out,
        good_count=good_count,
        bad_count=bad_count,
        orthogonalize_control_mean=not args.no_orthogonalize_control_mean,
        winsorization_quantile=args.winsorization_quantile,
        max_input_tokens=args.max_input_tokens,
        batch_size=args.batch_size,
        load_mode=load_mode,
        layer_policy=infer_layer_policy(layer_scales, QWEN_SPEC),
        activation_evidence=activation_evidence,
    )
    write_manifest_noreplace(args.output.resolve(), manifest, profile)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv if argv is not None else sys.argv[1:])
        manifest = run(args)
    except (InterventionError, qwen_quantize.PipelineError) as exc:
        print(f"qwen_intervention.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(args.output.resolve()),
        "direction_count": len(manifest["directions"]),
        "target_count": len(manifest["targets"]),
        "efficacy_evaluated": False,
        "publishes": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
