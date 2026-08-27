#!/usr/bin/env python3
"""Capture stock Qwen ROCMI4 activations and emit 16 intervention manifests.

The gfx1151 portion runs the exact digest-bound Ember image against the exact
stock-control shard set.  MTP is verified as companion provenance but is
explicitly disabled during capture: directions must come from stock ROCMI4.
Only the pinned OtherU 32 good and 32 bad extraction rows are submitted.  The
sweep and final partitions are never opened; their pinned digests are retained
only as selection and confirmation boundaries in generated evidence.

This tool never quantizes, evaluates, selects a policy, publishes, or uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence
import urllib.error
import urllib.request

import qwen_bakeoff
import qwen_intervention
import qwen_quantize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "share/release_profiles/qwen3.8-flash-next-rocmi4-strix-halo.json"
DEFAULT_CONTRACT = ROOT / "share/quant_eval/qwen3.8-otheru-corpus-contract.json"
DEFAULT_RECIPE = ROOT / "share/quant_eval/qwen3.8-flash-next-bakeoff.json"
GPU_LOCK = Path("/usr/local/sbin/ember-gpu-lock")
PRODUCTION = Path("/usr/local/sbin/ember-cert-production")
HEX64 = set("0123456789abcdef")
RECORD_BYTES = 48 * 2560 * 4
CONTAINER_LOG_NAME = "activation-container.log"


class CaptureError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direct_sha256(path: Path) -> str:
    """Hash large UMA artifacts through direct I/O, outside page cache."""
    process = subprocess.Popen(
        ["dd", f"if={path}", "iflag=direct", "bs=8M", "status=none"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
        digest.update(block)
    if process.wait() != 0:
        raise CaptureError(f"O_DIRECT integrity read failed: {path}")
    return digest.hexdigest()


def device_group_args() -> list[str]:
    """Return numeric supplemental groups for the exact mounted GPU nodes.

    ROCm development images do not necessarily define host group names such as
    ``render``. Docker accepts numeric GIDs, which preserve device access
    without depending on the container's ``/etc/group``.
    """
    nodes = [Path("/dev/kfd")]
    dri = Path("/dev/dri")
    if dri.is_dir():
        nodes.extend(sorted(dri.iterdir()))
    gids: set[int] = set()
    for node in nodes:
        try:
            info = node.stat()
        except OSError as exc:
            raise CaptureError(f"cannot inspect GPU device group: {node}: {exc}") from exc
        if stat.S_ISCHR(info.st_mode):
            gids.add(info.st_gid)
    if not gids:
        raise CaptureError("no character-device GIDs found for /dev/kfd or /dev/dri")
    result: list[str] = []
    for gid in sorted(gids):
        result.extend(("--group-add", str(gid)))
    return result


def require_digest(value: str, option: str, *, prefixed: bool = False) -> str:
    raw = value.removeprefix("sha256:") if prefixed else value
    expected_length = 71 if prefixed else 64
    if len(value) != expected_length or (prefixed and not value.startswith("sha256:")):
        raise CaptureError(f"{option} must be {'sha256: plus ' if prefixed else ''}64 lowercase hex characters")
    if len(raw) != 64 or any(character not in HEX64 for character in raw):
        raise CaptureError(f"{option} must be {'sha256: plus ' if prefixed else ''}64 lowercase hex characters")
    return value


def require_absolute(path: Path, option: str) -> Path:
    if not path.is_absolute():
        raise CaptureError(f"{option} must be an absolute path")
    return path


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"{label} must be a JSON object")
    return value


def run_checked(command: Sequence[str], *, capture: bool = False) -> str:
    try:
        result = subprocess.run(
            list(command), check=True, text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise CaptureError(f"command failed: {command[0]}{': ' + detail if detail else ''}") from exc
    return result.stdout.strip() if capture and result.stdout is not None else ""


def image_identity(image: str, expected_digest: str, output: Path) -> str:
    raw = run_checked(["docker", "image", "inspect", image], capture=True)
    output.write_text(raw + "\n", encoding="utf-8")
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CaptureError("docker returned invalid image inspection JSON") from exc
    if not isinstance(records, list) or len(records) != 1:
        raise CaptureError("Ember image reference did not resolve exactly once")
    record = records[0]
    observed = {record.get("Id")}
    observed.update(item.rsplit("@", 1)[-1] for item in record.get("RepoDigests") or [])
    if expected_digest not in observed:
        raise CaptureError(
            f"Ember image digest mismatch: expected {expected_digest}, observed {sorted(observed)}"
        )
    labels = ((record.get("Config") or {}).get("Labels") or {})
    revision = labels.get("org.opencontainers.image.revision", "")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in HEX64 for c in revision):
        revision = run_checked([
            "docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-c",
            "awk -F= '$1 == \"EMBER_CONFIGURED_GIT_HEAD:STRING\" { print $2 }' "
            "/ember/build-rocm/CMakeCache.txt",
        ], capture=True)
    if len(revision) != 40 or any(c not in HEX64 for c in revision):
        raise CaptureError("Ember image has no exact 40-hex source revision")
    return revision


def validate_stock_control(
    record_path: Path, record_sha: str, model: Path, model_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256(record_path) != record_sha:
        raise CaptureError("stock-control build-record digest mismatch")
    record = read_object(record_path, "stock-control build record")
    experiment = record.get("experiment") or {}
    if (
        record.get("status") != "complete"
        or record.get("mode") != "execute"
        or record.get("compute_mode") != "exact_dequant"
        or record.get("w4a4_enabled") is not False
        or experiment.get("kind") != "stock_control"
        or experiment.get("stock_weights_unchanged") is not True
        or experiment.get("final_release_eligible") is not False
        or record.get("intervention") is not None
    ):
        raise CaptureError("model build record is not the final-ineligible stock ROCMI4 control")
    build_info = ((record.get("tools") or {}).get("quantizer_build_info") or {})
    if build_info.get("format") != "Q4_0_ROCMI4" or build_info.get("ggml_tensor_type") != 108:
        raise CaptureError("stock control was not encoded as exact Q4_0_ROCMI4")
    rows = (record.get("output") or {}).get("shards")
    if not isinstance(rows, list) or not rows:
        raise CaptureError("stock-control build record has no ordered shard inventory")
    normalized: list[dict[str, Any]] = []
    recorded_parent: Path | None = None
    filenames: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CaptureError("stock-control shard inventory is malformed")
        recorded_path = Path(str(row.get("path", "")))
        # qwen-convert-control runs inside the dev image, where the persistent
        # host workspace is mounted at /qwen-work.  The build record therefore
        # carries container-absolute paths.  Bind that immutable inventory to
        # the explicitly supplied host model directory by basename; never
        # trust or rewrite an arbitrary parent from the record.
        path = model.parent / recorded_path.name
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if (
            not recorded_path.is_absolute() or recorded_path.name in filenames
            or path.is_symlink() or not path.is_file()
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in HEX64 for c in digest)
            or not isinstance(size, int) or isinstance(size, bool) or size < 1
            or path.stat().st_size != size
        ):
            raise CaptureError(f"stock-control shard {index} is missing or malformed")
        filenames.add(recorded_path.name)
        if recorded_parent is None:
            recorded_parent = recorded_path.parent
        elif recorded_path.parent != recorded_parent:
            raise CaptureError("stock-control record shards do not share one directory")
        normalized.append({"path": path, "recorded_path": recorded_path,
                           "sha256": digest, "size_bytes": size})
    if model.resolve() != normalized[0]["path"].resolve():
        raise CaptureError("--model must name the first ordered stock-control shard")
    if model_sha != normalized[0]["sha256"]:
        raise CaptureError("--model-sha256 differs from the stock-control record")
    return record, normalized


def validate_corpora(
    contract_path: Path, contract_sha: str, corpus_dir: Path,
    good_sha: str, bad_sha: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    if sha256(contract_path) != contract_sha:
        raise CaptureError("pinned OtherU corpus-contract digest mismatch")
    contract = read_object(contract_path, "OtherU corpus contract")
    source = contract.get("source") or {}
    if (
        source.get("repository") != "https://git.otheru.ai/akadmin/otheru-quant-pipeline"
        or source.get("revision") != "a3c6a728510f91394e991504951ac316cd3a89af"
        or contract.get("pairwise_request_overlap_count") != 0
    ):
        raise CaptureError("corpus contract is not the pinned disjoint OtherU contract")
    artifacts = contract.get("derived_artifacts") or {}
    good_row = artifacts.get("extraction-good.jsonl") or {}
    bad_row = artifacts.get("extraction-bad.jsonl") or {}
    if (
        good_row.get("record_count") != 32 or bad_row.get("record_count") != 32
        or good_row.get("sha256") != good_sha or bad_row.get("sha256") != bad_sha
    ):
        raise CaptureError("extraction corpus digests/counts differ from the pinned 32+32 contract")
    good = corpus_dir / "extraction-good.jsonl"
    bad = corpus_dir / "extraction-bad.jsonl"
    if good.is_symlink() or bad.is_symlink() or not good.is_file() or not bad.is_file():
        raise CaptureError("extraction corpora must be regular non-symlink files")
    if sha256(good) != good_sha or sha256(bad) != bad_sha:
        raise CaptureError("extraction corpus file digest mismatch")
    return good, bad, contract, artifacts


def iter_request_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CaptureError(f"invalid extraction JSONL row {line_number}: {path}") from exc
            record = qwen_intervention.parse_corpus_record(row, f"{path}:{line_number}")
            yield {"model": "qwen3.8-flash-next", "messages": list(record.messages),
                   "max_tokens": 0, "temperature": 0, "stream": False}


def request_capture(port: int, body: dict[str, Any]) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = response.read()
            if response.status != 200:
                raise CaptureError(f"activation request returned HTTP {response.status}")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CaptureError(f"activation request failed: {exc}") from exc
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CaptureError("activation request returned invalid JSON") from exc
    if not isinstance(parsed, dict) or "error" in parsed:
        raise CaptureError(f"activation request returned an error: {parsed}")


class ExclusiveGPU:
    def __init__(self, state_dir: Path | None = None) -> None:
        self.locked = False
        self.masked = False
        self.restore_service = False
        self.state_dir = state_dir

    def marker(self, name: str) -> Path | None:
        return self.state_dir / name if self.state_dir is not None else None

    def mark(self, name: str) -> None:
        path = self.marker(name)
        if path is None:
            return
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        qwen_quantize.fsync_directory(path.parent)

    def clear_mark(self, name: str) -> None:
        path = self.marker(name)
        if path is None:
            return
        path.unlink(missing_ok=True)
        qwen_quantize.fsync_directory(path.parent)

    @staticmethod
    def sudo(wrapper: Path, action: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sudo", "-n", str(wrapper), action], check=check, text=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def acquire(self) -> None:
        self.sudo(GPU_LOCK, "acquire")
        self.locked = True
        self.mark(".gpu-lock-held")
        if self.sudo(PRODUCTION, "is-active", check=False).returncode == 0:
            self.restore_service = True
            self.mark(".production-was-active")
            self.sudo(PRODUCTION, "stop")
        self.sudo(PRODUCTION, "mask")
        self.masked = True
        self.mark(".production-masked")

    @staticmethod
    def retry(wrapper: Path, action: str) -> bool:
        for _ in range(3):
            if ExclusiveGPU.sudo(wrapper, action, check=False).returncode == 0:
                return True
            time.sleep(1)
        return False

    def restore(self) -> None:
        failures: list[str] = []
        if self.masked:
            if self.retry(PRODUCTION, "unmask"):
                self.masked = False
                self.clear_mark(".production-masked")
            else:
                failures.append("unmask production")
        if self.restore_service:
            if not self.masked and self.retry(PRODUCTION, "start"):
                self.restore_service = False
                self.clear_mark(".production-was-active")
            else:
                failures.append("restart production")
        if self.locked:
            if self.retry(GPU_LOCK, "release"):
                self.locked = False
                self.clear_mark(".gpu-lock-held")
            else:
                failures.append("release GPU lock")
        if failures:
            raise CaptureError("restore failed: " + ", ".join(failures))


def remove_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def retain_container_logs(name: str, output_dir: Path) -> Path:
    """Durably retain combined container output without replacing any path.

    The output directory is opened without following a final symlink, then the
    fixed log basename is created relative to that directory descriptor with
    O_EXCL. This keeps a concurrent file or symlink from redirecting or being
    overwritten by failure forensics.
    """
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory_fd = os.open(output_dir, directory_flags)
    except OSError as exc:
        raise CaptureError(f"cannot open capture output directory for container logs: {exc}") from exc
    log_fd = -1
    try:
        try:
            log_fd = os.open(
                CONTAINER_LOG_NAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError as exc:
            raise CaptureError(
                f"refusing to overwrite container log: {output_dir / CONTAINER_LOG_NAME}"
            ) from exc
        except OSError as exc:
            raise CaptureError(f"cannot create container log: {exc}") from exc
        try:
            result = subprocess.run(
                ["docker", "logs", "--timestamps", name], check=False,
                stdout=log_fd, stderr=subprocess.STDOUT, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CaptureError(f"cannot retain container logs: {exc}") from exc
        finally:
            os.fsync(log_fd)
        if result.returncode != 0:
            raise CaptureError(
                f"docker logs exited {result.returncode}; diagnostic output was retained"
            )
    finally:
        try:
            if log_fd >= 0:
                os.fsync(directory_fd)
        finally:
            if log_fd >= 0:
                os.close(log_fd)
            os.close(directory_fd)
    return output_dir / CONTAINER_LOG_NAME


def cleanup_capture_container(
    name: str, output_dir: Path, started: bool,
    capture_error: BaseException | None,
) -> BaseException | None:
    """Retain failure logs before deleting an owned capture container."""
    if started and capture_error is not None:
        try:
            retain_container_logs(name, output_dir)
        except BaseException as log_exc:
            capture_error = CaptureError(
                f"{capture_error}; additionally failed to retain container logs: {log_exc}"
            )
    remove_container(name)
    return capture_error


def wait_healthy(port: int, container: str) -> None:
    endpoint = f"http://127.0.0.1:{port}/health"
    for _ in range(360):
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            pass
        state = run_checked(
            ["docker", "inspect", "--format", "{{.State.Running}}", container],
            capture=True,
        )
        if state != "true":
            raise CaptureError("activation-capture container exited during model load")
        time.sleep(5)
    raise CaptureError("activation-capture server did not become healthy within 30 minutes")


def split_dump(combined: Path, good: Path, bad: Path) -> None:
    expected = 64 * RECORD_BYTES
    if combined.is_symlink() or not combined.is_file() or combined.stat().st_size != expected:
        raise CaptureError(f"activation capture has {combined.stat().st_size if combined.exists() else 0} bytes; expected {expected}")
    half = 32 * RECORD_BYTES
    with combined.open("rb", buffering=0) as source:
        for output in (good, bad):
            with output.open("xb", buffering=0) as destination:
                remaining = half
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise CaptureError("activation capture ended while splitting exact rows")
                    destination.write(block)
                    remaining -= len(block)
                destination.flush()
                os.fsync(destination.fileno())
        if source.read(1):
            raise CaptureError("activation capture grew while splitting")


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CaptureError(f"refusing to overwrite output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def intervention_grid(recipe: dict[str, Any]) -> list[tuple[str, float, str, list[float]]]:
    """Return the exact deterministic 4x4 policy grid, never measurements."""
    lambdas = recipe.get("intervention_sweep", {}).get("lambdas")
    policies = recipe.get("intervention_sweep", {}).get("layer_policies")
    if lambdas != [0.25, 0.5, 0.75, 1.0] or list(policies or {}) != [
        "band-10-42", "upper-24", "upper-12", "non-qsa-band-10-42"
    ]:
        raise CaptureError("recipe does not define the exact 4x4 intervention policy grid")
    result: list[tuple[str, float, str, list[float]]] = []
    for scale in lambdas:
        for policy in policies:
            selected = set(qwen_bakeoff.layers(policy))
            values = [float(scale) if layer in selected else 0.0 for layer in range(48)]
            result.append((f"lambda-{float(scale):.2f}-{policy}", float(scale), policy, values))
    if len(result) != 16 or len({row[0] for row in result}) != 16:
        raise CaptureError("intervention recipe did not expand to 16 unique policies")
    return result


def generate_manifests(
    args: argparse.Namespace, good_corpus: Path, bad_corpus: Path,
    contract: dict[str, Any], artifacts: dict[str, Any], model_sha: str,
    good_dump: Path, bad_dump: Path, image_revision: str,
) -> list[dict[str, Any]]:
    profile, _inventory, _path = qwen_quantize.validate_profile(args.profile.resolve())
    recipe = read_object(args.recipe, "intervention bakeoff recipe")
    if sha256(args.recipe) != args.recipe_sha256:
        raise CaptureError("intervention recipe digest mismatch")
    grid = intervention_grid(recipe)
    good = qwen_intervention.scan_corpus(
        good_corpus, "qwen-otheru-extraction-good-32", max_records=32, max_line_bytes=1024 * 1024,
    )
    bad = qwen_intervention.scan_corpus(
        bad_corpus, "qwen-otheru-extraction-bad-32", max_records=32, max_line_bytes=1024 * 1024,
    )
    if good.record_count != 32 or bad.record_count != 32:
        raise CaptureError("direction extraction requires exactly 32 good and 32 bad rows")
    sweep = artifacts.get("sweep-validation.jsonl") or {}
    final = artifacts.get("final-heldout.jsonl") or {}
    if sweep.get("record_count") != 134 or final.get("record_count") != 134:
        raise CaptureError("pinned selection/confirmation partition metadata is incomplete")
    held_out = qwen_intervention.CorpusEvidence(
        path=Path("selection-only-not-opened"),
        corpus_id="qwen-otheru-sweep-validation-selection-only",
        sha256=sweep["sha256"], record_count=sweep["record_count"],
        content_fingerprints=frozenset(),
    )
    good_means, good_dump_sha = qwen_intervention.activation_dump_means(
        good_dump, expected_records=32, spec=qwen_intervention.QWEN_SPEC,
        winsorization_quantile=1.0,
    )
    bad_means, bad_dump_sha = qwen_intervention.activation_dump_means(
        bad_dump, expected_records=32, spec=qwen_intervention.QWEN_SPEC,
        winsorization_quantile=1.0,
    )
    directions = qwen_intervention.build_directions(
        good_means, bad_means, orthogonalize_control_mean=True,
        spec=qwen_intervention.QWEN_SPEC,
    )
    activation_evidence = {
        "backend": "ember_qwen_runtime_f32_dump",
        "format": "48x2560-little-endian-f32-writer-output-records-v2",
        "record_order": "corpus_jsonl_order",
        "stock_rocmi4_artifact_sha256": model_sha,
        "artifact_sha256_verification": "supplied_not_locally_rehashed",
        "good_dump_sha256": good_dump_sha,
        "bad_dump_sha256": bad_dump_sha,
        "good_dump_bytes": good_dump.stat().st_size,
        "bad_dump_bytes": bad_dump.stat().st_size,
        "ember_image_revision": image_revision,
        "mtp_companion_sha256": args.mtp_sha256,
        "mtp_loaded_during_capture": False,
    }
    manifests_dir = args.output_dir / "interventions"
    manifests_dir.mkdir()
    generated: list[dict[str, Any]] = []
    for identifier, scale, policy, layer_scales in grid:
        manifest = qwen_intervention.build_manifest(
            profile=profile, directions=directions, layer_scales=layer_scales,
            good=good, bad=bad, held_out=held_out, good_count=32, bad_count=32,
            orthogonalize_control_mean=True, winsorization_quantile=1.0,
            max_input_tokens=0, batch_size=1,
            load_mode="stock_rocmi4_runtime_f32_dump",
            layer_policy=policy,
            activation_evidence=activation_evidence,
        )
        manifest["selection_partition_policy"] = {
            "direction_inputs": [
                {"id": good.corpus_id, "sha256": good.sha256, "record_count": 32},
                {"id": bad.corpus_id, "sha256": bad.sha256, "record_count": 32},
            ],
            "sweep_validation": {
                "sha256": sweep["sha256"], "record_count": 134,
                "usage": "selection_only_not_read_during_direction_generation",
            },
            "final_heldout": {
                "sha256": final["sha256"], "record_count": 134,
                "usage": "single_winner_confirmation_only_not_read_during_direction_generation",
            },
            "final_may_select_recipe": False,
            "pairwise_request_overlap_count": contract["pairwise_request_overlap_count"],
        }
        output = manifests_dir / f"{identifier}.json"
        qwen_intervention.write_manifest_noreplace(output, manifest, profile)
        generated.append({
            "id": identifier, "lambda": scale, "layer_policy": policy,
            "filename": f"interventions/{output.name}", "sha256": sha256(output),
            "target_count": len(manifest["targets"]),
        })
    if len(generated) != 16:
        raise CaptureError("intervention generation did not produce all 16 policies")
    return generated


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-revision", required=True)
    parser.add_argument("--artifact-revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--binary", default="/usr/local/bin/ember-dflash")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--control-record", type=Path, required=True)
    parser.add_argument("--control-record-sha256", required=True)
    parser.add_argument("--mtp", type=Path, required=True)
    parser.add_argument("--mtp-sha256", required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--corpus-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--corpus-contract-sha256", required=True)
    parser.add_argument("--good-corpus-sha256", required=True)
    parser.add_argument("--bad-corpus-sha256", required=True)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18087)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv))


def validate_syntax(args: argparse.Namespace) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", args.tool_revision) is None:
        raise CaptureError("--tool-revision must be a full lowercase Ember commit")
    if re.fullmatch(r"[0-9a-f]{40}", args.artifact_revision) is None:
        raise CaptureError("--artifact-revision must be a full lowercase Ember commit")
    require_digest(args.image_digest, "--image-digest", prefixed=True)
    for name in (
        "model_sha256", "control_record_sha256", "mtp_sha256",
        "corpus_contract_sha256", "good_corpus_sha256", "bad_corpus_sha256",
        "recipe_sha256",
    ):
        require_digest(getattr(args, name), "--" + name.replace("_", "-"))
    for name in (
        "model", "control_record", "mtp", "corpus_dir", "corpus_contract",
        "recipe", "profile", "output_dir",
    ):
        require_absolute(getattr(args, name), "--" + name.replace("_", "-"))
    if not args.binary.startswith("/"):
        raise CaptureError("--binary must be an absolute in-image path")
    if not 1024 <= args.port <= 65535:
        raise CaptureError("--port must be 1024..65535")


def print_plan(args: argparse.Namespace) -> None:
    print("plan:")
    print(f"  capture tooling   {args.tool_revision}")
    print(f"  artifact tooling  {args.artifact_revision}")
    print(f"  Ember image       {args.image}")
    print(f"  image digest      {args.image_digest}")
    print(f"  stock model       {args.model}")
    print(f"  model sha256      {args.model_sha256}")
    print(f"  MTP inventory     {args.mtp} ({args.mtp_sha256}); verified, not loaded")
    print(f"  corpus contract   {args.corpus_contract_sha256}")
    print("  extraction        exact pinned OtherU 32 good + 32 bad rows")
    print("  sweep/final       digest metadata only; selection/confirmation, never direction inputs")
    print("  output            48x2560 writer-output directions; deterministic 4 lambdas x 4 policies")
    print(f"  exclusive GPU     {GPU_LOCK} + {PRODUCTION} stop/mask/restore")
    print("  publication       none")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    for command in ("docker", "sudo", "dd"):
        if shutil.which(command) is None:
            raise CaptureError(f"{command} is required")
    if not os.access(GPU_LOCK, os.X_OK) or not os.access(PRODUCTION, os.X_OK):
        raise CaptureError("fixed-purpose GPU lock or production wrapper is missing")
    if not Path("/dev/kfd").is_char_device() or not Path("/dev/dri").is_dir():
        raise CaptureError("activation capture must run on the gfx1151 host")
    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=2):
            raise CaptureError(f"port {args.port} is already in use")
    except (ConnectionRefusedError, TimeoutError, OSError):
        pass
    kernel_options = Path("/proc/cmdline").read_text(encoding="utf-8").split()
    if "iommu=off" in kernel_options or "amd_iommu=off" in kernel_options:
        raise CaptureError("IOMMU is disabled")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise CaptureError("--output-dir must not exist")
    for path, label in (
        (args.model, "model"), (args.control_record, "control record"),
        (args.mtp, "MTP"), (args.corpus_contract, "corpus contract"),
        (args.recipe, "recipe"), (args.profile, "profile"),
    ):
        if path.is_symlink() or not path.is_file():
            raise CaptureError(f"{label} must be a regular non-symlink file: {path}")
    args.output_dir.mkdir(parents=True)
    image_revision = image_identity(
        args.image, args.image_digest, args.output_dir / "image-inspect.json"
    )
    if image_revision != args.tool_revision:
        raise CaptureError("capture tool and runtime image revisions differ")
    run_checked([
        "docker", "run", "--rm", "--entrypoint", "/bin/sh", args.image,
        "-c", 'test -x "$1"', "qwen-capture-control-preflight", args.binary,
    ])
    record, shards = validate_stock_control(
        args.control_record, args.control_record_sha256, args.model, args.model_sha256,
    )
    record_revision = ((record.get("tools") or {}).get("quantizer_build_info") or {}).get(
        "ember_revision"
    )
    if record_revision != args.artifact_revision:
        raise CaptureError("stock-control model revision does not match --artifact-revision")
    for row in shards:
        if direct_sha256(row["path"]) != row["sha256"]:
            raise CaptureError(f"stock-control shard digest mismatch: {row['path']}")
    if direct_sha256(args.mtp) != args.mtp_sha256:
        raise CaptureError("MTP companion digest mismatch")
    good_corpus, bad_corpus, contract, artifacts = validate_corpora(
        args.corpus_contract, args.corpus_contract_sha256, args.corpus_dir,
        args.good_corpus_sha256, args.bad_corpus_sha256,
    )
    combined = args.output_dir / "activations-64x48x2560.f32"
    container = f"qwen-capture-control-{os.getpid()}"
    # Durable ownership markers let the outer certification workflow recover
    # after a runner-level kill, while successful in-process cleanup removes
    # every marker.  Recovery therefore never releases another job's lock.
    exclusive = ExclusiveGPU(args.output_dir)
    capture_error: BaseException | None = None
    container_started = False

    def interrupted(signum: int, _frame: Any) -> None:
        raise CaptureError(f"interrupted by signal {signum}")

    previous_handlers = {
        signum: signal.signal(signum, interrupted)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        exclusive.acquire()
        run_checked([
            "docker", "run", "-d", "--name", container, "--network", "host",
            "--device", "/dev/kfd", "--device", "/dev/dri",
            *device_group_args(), "--ipc", "host",
            "--security-opt", "seccomp=unconfined", "--ulimit", "memlock=-1:-1",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{args.model.parent}:/control:ro",
            "-v", f"{args.output_dir}:/capture",
            "-e", "DFLASH_QWEN_ACT_DUMP=/capture/activations-64x48x2560.f32",
            "-e", "DFLASH_QWEN_MTP=", "-e", "DFLASH_QWEN_MTP_DEPTH=",
            "-e", "EMBER_IDLE_RECLAIM_SECS=0",
            "--entrypoint", args.binary, args.image,
            "-m", f"/control/{args.model.name}", "--host", "127.0.0.1",
            "--port", str(args.port), "--max-ctx", "8192", "--prefix-cache-slots", "1",
        ])
        container_started = True
        wait_healthy(args.port, container)
        row_count = 0
        for corpus in (good_corpus, bad_corpus):
            for body in iter_request_rows(corpus):
                request_capture(args.port, body)
                row_count += 1
                actual = combined.stat().st_size if combined.is_file() else 0
                expected = row_count * RECORD_BYTES
                if actual != expected:
                    raise CaptureError(
                        f"activation row {row_count} missing/truncated: {actual} bytes, expected {expected}"
                    )
        if row_count != 64:
            raise CaptureError(f"activation capture processed {row_count} rows, expected exactly 64")
    except BaseException as exc:  # preserve cleanup across command and signal failures
        capture_error = exc
    finally:
        # A second termination signal must not interrupt the restore sequence.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        capture_error = cleanup_capture_container(
            container, args.output_dir, container_started, capture_error,
        )
        try:
            exclusive.restore()
        except BaseException as restore_exc:
            capture_error = restore_exc if capture_error is None else CaptureError(
                f"{capture_error}; additionally {restore_exc}"
            )
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if capture_error is not None:
        raise capture_error
    (args.output_dir / "activations-64x48x2560.f32.lock").unlink(missing_ok=True)

    good_dump = args.output_dir / "extraction-good-32x48x2560.f32"
    bad_dump = args.output_dir / "extraction-bad-32x48x2560.f32"
    split_dump(combined, good_dump, bad_dump)
    generated = generate_manifests(
        args, good_corpus, bad_corpus, contract, artifacts, args.model_sha256,
        good_dump, bad_dump, image_revision,
    )
    evidence = {
        "schema": "ember.qwen3.8.stock-control-activation-capture.v1",
        "status": "complete",
        "publishes": False,
        "stock_rocmi4_only": True,
        "capture_tool": {"ember_revision": args.tool_revision},
        "image": {"ref": args.image, "digest": args.image_digest,
                  "ember_revision": image_revision},
        "model": {
            "primary_path": args.model.name, "primary_sha256": args.model_sha256,
            "build_record_sha256": args.control_record_sha256,
            "quantizer_ember_revision": record_revision,
            "shards": [{"filename": row["path"].name, "sha256": row["sha256"],
                        "size_bytes": row["size_bytes"]} for row in shards],
        },
        "mtp": {"filename": args.mtp.name, "sha256": args.mtp_sha256,
                "verified": True, "loaded_during_capture": False},
        "corpus": {
            "contract_sha256": args.corpus_contract_sha256,
            "source_revision": contract["source"]["revision"],
            "extraction_good_sha256": args.good_corpus_sha256,
            "extraction_bad_sha256": args.bad_corpus_sha256,
            "rows": {"good": 32, "bad": 32},
            "sweep_usage": "selection_only_not_read",
            "final_usage": "single_winner_confirmation_only_not_read",
        },
        "activations": {
            "format": "48x2560-little-endian-f32-writer-output-records-v2",
            "combined_sha256": sha256(combined),
            "good_sha256": sha256(good_dump), "bad_sha256": sha256(bad_dump),
            "direction_shape": [48, 2560],
        },
        "recipe_sha256": args.recipe_sha256,
        "interventions": generated,
    }
    write_json_new(args.output_dir / "capture-manifest.json", evidence)
    qwen_quantize.fsync_directory(args.output_dir)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv if argv is not None else sys.argv[1:])
        validate_syntax(args)
        print_plan(args)
        if args.dry_run:
            print("qwen_capture_control.py: dry run; no files, Docker, GPU, sudo, or production state touched")
            return 0
        evidence = execute(args)
    except (CaptureError, qwen_intervention.InterventionError,
            qwen_quantize.PipelineError, OSError) as exc:
        print(f"qwen_capture_control.py: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": evidence["status"], "publishes": False,
                      "manifest": str((args.output_dir / 'capture-manifest.json').resolve()),
                      "interventions": len(evidence["interventions"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
