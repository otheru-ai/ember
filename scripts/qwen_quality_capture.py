#!/usr/bin/env python3
"""Capture schema-v2 Qwen quality evidence from live Ember containers.

The four subcommands are deliberately staged because a 128 GiB Strix Halo
cannot keep the stock, candidate, and judge models resident together.  Each
capture phase talks to a live HTTP endpoint, inspects that exact running Docker
container, and writes new files without overwriting prior evidence.  ``assemble``
only joins phase records produced by this exact driver and immutable plan.

No bearer tokens, request headers, or arbitrary container environment values
are written.  Only the target-only settings needed by the verifier are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Sequence
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
IMAGE_REFERENCE = re.compile(r"[^\s@]+@sha256:([0-9a-f]{64})")
TARGET_ENV = (
    "DFLASH_DS4_SPEC", "DFLASH_DSPARK_XDNA_PLUGIN",
    "DFLASH_DSPARK_XDNA_GPU_MAIN", "DFLASH_DSPARK_XDNA_REQUIRED",
)
SUBMIT_VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Submit the blind paired quality verdict.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "required": ["ratings", "preference", "severity"],
            "properties": {
                "ratings": {
                    "type": "object", "additionalProperties": False,
                    "required": ["A", "B"],
                    "properties": {
                        side: {
                            "type": "object", "additionalProperties": False,
                            "required": ["engagement", "coherence", "correctness", "cited_spans"],
                            "properties": {
                                "engagement": {"enum": ["deflection", "invalid", "partial", "refusal", "substantive"]},
                                "coherence": {"type": "integer", "minimum": 0, "maximum": 4},
                                "correctness": {"type": "integer", "minimum": 0, "maximum": 4},
                                "cited_spans": {"type": "array", "minItems": 1,
                                                 "items": {"type": "string", "minLength": 8}},
                            },
                        } for side in ("A", "B")
                    },
                },
                "preference": {"enum": ["A", "tie", "B"]},
                "severity": {"enum": ["none", "minor", "major"]},
            },
        },
    },
}


class CaptureError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def executing_driver_sha256() -> str:
    return digest_file(Path(__file__).resolve())[0]


def digest_file(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"artifact must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    count = 0
    if path.stat().st_size >= 512 * 1024 * 1024:
        process = subprocess.Popen(
            ["dd", f"if={path}", "iflag=direct", "bs=8M", "status=none"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(block)
            count += len(block)
        process.stdout.close()
        assert process.stderr is not None
        detail = process.stderr.read().decode("utf-8", "replace").strip()
        process.stderr.close()
        if process.wait() != 0:
            raise CaptureError(f"direct-I/O artifact hash failed: {path}: {detail}")
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                count += len(block)
    return digest.hexdigest(), count


def write_new(path: Path, payload: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"path": str(path), "sha256": digest_bytes(payload)}


def write_json(path: Path, value: Any) -> dict[str, str]:
    return write_new(path, canonical(value))


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    return write_new(path, b"".join(canonical(row) for row in rows))


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read {label}: {exc}") from exc


def pinned(path: Path, expected: str, label: str) -> bytes:
    if HEX64.fullmatch(expected) is None:
        raise CaptureError(f"{label} SHA-256 is malformed")
    data = path.read_bytes()
    if digest_bytes(data) != expected:
        raise CaptureError(f"{label} SHA-256 mismatch")
    return data


def command(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(list(args), check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or ""
        raise CaptureError(f"command failed ({args[0]}): {detail.strip()}") from exc
    return result.stdout


def require_plan(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    plan = json.loads(data)
    if not isinstance(plan, dict) or plan.get("schema") != "ember.qwen3.8.quality-capture-plan.v1":
        raise CaptureError("unsupported quality capture plan")
    required = {"schema", "contract_id", "corpus", "rubric", "agentic_cases",
                "models", "judge", "runs"}
    if set(plan) != required:
        raise CaptureError("quality capture plan keys differ")
    return plan, digest_bytes(data)


def descriptor_path(plan_path: Path, descriptor: dict[str, Any], label: str) -> Path:
    if set(descriptor) < {"path", "sha256"}:
        raise CaptureError(f"{label} descriptor is incomplete")
    path = Path(descriptor["path"])
    if not path.is_absolute():
        raise CaptureError(f"{label} path must be absolute so the assembled contract is relocatable")
    return path


def inventory(descriptor: dict[str, Any], plan_path: Path, label: str) -> str:
    shards = descriptor.get("shards")
    if not isinstance(shards, list) or not shards:
        raise CaptureError(f"{label} has no ordered shards")
    identities = []
    for index, row in enumerate(shards, 1):
        path = descriptor_path(plan_path, row, f"{label} shard {index}")
        actual_sha, actual_bytes = digest_file(path)
        if row.get("sha256") != actual_sha or row.get("bytes") != actual_bytes:
            raise CaptureError(f"{label} shard {index} identity mismatch")
        identities.append({"index": index, "sha256": actual_sha, "bytes": actual_bytes})
    return digest_bytes(canonical(identities))


def artifact(descriptor: dict[str, Any], plan_path: Path, label: str) -> tuple[Path, str]:
    path = descriptor_path(plan_path, descriptor, label)
    actual_sha, actual_bytes = digest_file(path)
    if descriptor.get("sha256") != actual_sha or descriptor.get("bytes") != actual_bytes:
        raise CaptureError(f"{label} identity mismatch")
    return path, actual_sha


def request(endpoint: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise CaptureError("capture endpoints must be loopback HTTP URLs")
    wire = canonical(body)
    call = urllib.request.Request(endpoint, data=wire,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(call, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CaptureError(f"runtime request failed: {exc}") from exc
    if status != 200:
        raise CaptureError(f"runtime request returned HTTP {status}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CaptureError("runtime returned non-JSON output") from exc
    if not isinstance(value, dict) or "error" in value:
        raise CaptureError(f"runtime returned an error: {value}")
    return value


def inspect_runtime(
    run: dict[str, Any], output: Path, role: str, loaded_identity: str,
    loaded_host_paths: Sequence[Path],
) -> dict[str, Any]:
    if set(run) != {"container", "endpoint", "image", "loaded_path"}:
        raise CaptureError(f"{role} run definition keys differ")
    container = run["container"]
    raw_container = command(["docker", "container", "inspect", container])
    records = json.loads(raw_container)
    if not isinstance(records, list) or len(records) != 1:
        raise CaptureError(f"{role} container did not resolve exactly once")
    record = records[0]
    if (record.get("State") or {}).get("Running") is not True:
        raise CaptureError(f"{role} container is not running")
    container_id = record.get("Id")
    image_id = record.get("Image")
    if HEX64.fullmatch(str(container_id)) is None or re.fullmatch(r"sha256:[0-9a-f]{64}", str(image_id)) is None:
        raise CaptureError(f"{role} container identity is malformed")
    raw_image = command(["docker", "image", "inspect", image_id])
    image_records = json.loads(raw_image)
    if not isinstance(image_records, list) or len(image_records) != 1:
        raise CaptureError(f"{role} image did not resolve exactly once")
    image_record = image_records[0]
    expected_image = run["image"]
    match = IMAGE_REFERENCE.fullmatch(expected_image) if isinstance(expected_image, str) else None
    repo_digests = image_record.get("RepoDigests") or []
    if match is None or expected_image not in repo_digests or image_record.get("Id") != image_id:
        raise CaptureError(f"{role} immutable image identity mismatch")
    revision = (((image_record.get("Config") or {}).get("Labels") or {})
                .get("org.opencontainers.image.revision"))
    if not isinstance(revision, str) or HEX40.fullmatch(revision) is None:
        raise CaptureError(f"{role} image has no exact source revision label")
    # Read only the four policy variables from PID 1.  Reading `docker inspect`
    # Config.Env would miss entrypoint exports; dumping all of /proc/1/environ
    # would unnecessarily ingest credentials.
    env = {}
    for key in TARGET_ENV:
        env[key] = command([
            "docker", "exec", container, "/bin/sh", "-c",
            "awk -v key=\"$1\" 'BEGIN { RS=\"\\0\" } index($0,key\"=\")==1 "
            "{ sub(\"^[^=]*=\",\"\"); print; found=1 } END { if (!found) print \"\" }' "
            "/proc/1/environ",
            "qwen-quality-env", key,
        ]).rstrip("\n")
    if env["DFLASH_DS4_SPEC"] not in ("", "0") or any(env[key] for key in TARGET_ENV[1:]):
        raise CaptureError(f"{role} is not a non-speculative target-only runtime")
    agents = sorted(set(re.findall(r"\bgfx[0-9a-f]+\b",
        command(["docker", "exec", container, "rocminfo"]))))
    if "gfx1151" not in agents:
        raise CaptureError(f"{role} runtime did not observe gfx1151")
    process = [str(record.get("Path", "")), *[str(value) for value in record.get("Args") or []]]
    loaded_path = run["loaded_path"]
    if not isinstance(loaded_path, str) or not loaded_path.startswith("/") or loaded_path not in process:
        raise CaptureError(f"{role} process is not bound to the declared loaded artifact")
    if not loaded_host_paths or any(path.parent != loaded_host_paths[0].parent
                                    for path in loaded_host_paths):
        raise CaptureError(f"{role} loaded artifacts must share one host directory")
    mount_source = str(loaded_host_paths[0].parent.resolve())
    mount_target = str(Path(loaded_path).parent)
    matching_mounts = [row for row in record.get("Mounts") or []
                       if row.get("Source") == mount_source
                       and row.get("Destination") == mount_target
                       and row.get("RW") is False]
    if len(matching_mounts) != 1:
        raise CaptureError(f"{role} artifact directory is not mounted read-only at the loaded path")
    binary_path = output / "runtime" / f"{role}-ember-dflash"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    if binary_path.exists():
        raise CaptureError(f"refusing to overwrite runtime binary: {binary_path}")
    command(["docker", "cp", f"{container}:/usr/local/bin/ember-dflash", str(binary_path)])
    binary_sha, binary_bytes = digest_file(binary_path)
    inspection = {
        "schema": "ember.qwen3.8.docker-runtime-inspection.v1",
        "role": role,
        "container_id": container_id,
        "container_name": container,
        "endpoint": run["endpoint"],
        "running": True,
        "image": expected_image,
        "image_digest": "sha256:" + match.group(1),
        "image_id": image_id,
        "engine_revision": revision,
        "target_environment": env,
        "process_argv": process,
        "loaded_path": loaded_path,
        "loaded_mount": {"source": mount_source, "destination": mount_target, "read_only": True},
        "loaded_artifact_identity_sha256": loaded_identity,
        "device_agents": agents,
        "server_binary_sha256": binary_sha,
    }
    inspection_desc = write_json(output / "runtime" / f"{role}-inspection.json", inspection)
    return {
        "image": expected_image, "image_digest": "sha256:" + match.group(1),
        "image_id": image_id, "engine_revision": revision,
        "server_binary": {"path": str(binary_path), "sha256": binary_sha, "bytes": binary_bytes},
        "device_architecture": "gfx1151", "target_only": True,
        "speculative_decode": False, "inspection": inspection_desc,
    }


def load_rows(plan_path: Path, descriptor: dict[str, Any], label: str) -> tuple[list[dict[str, Any]], str]:
    path = descriptor_path(plan_path, descriptor, label)
    data = pinned(path, descriptor["sha256"], label)
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CaptureError(f"{label}:{number} is not an object")
            rows.append(value)
    return rows, descriptor["sha256"]


def response_body(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected") or {}
    maximum = expected.get("max_tokens", 512)
    return {"model": "qwen3.8-flash-next", "messages": case["messages"],
            "max_tokens": maximum, "temperature": 0, "seed": 7301, "stream": False}


def agentic_body(case: dict[str, Any]) -> dict[str, Any]:
    body = {"model": "qwen3.8-flash-next", "messages": case["messages"],
            "max_tokens": case["max_tokens"], "temperature": 0,
            "seed": 7301, "stream": False}
    if case.get("tools"):
        body["tools"] = case["tools"]
        body["tool_choice"] = "auto"
    return body


def model_inventory_digest(model: dict[str, Any]) -> str:
    rows = [{"index": index, "sha256": row["sha256"], "bytes": row["bytes"]}
            for index, row in enumerate(model["shards"], 1)]
    return digest_bytes(canonical(rows))


def capture_variant(args: argparse.Namespace, *, agentic: bool) -> None:
    plan, plan_sha = require_plan(args.plan)
    variant = args.variant
    model = plan["models"][variant]
    model_identity = inventory(model, args.plan, f"{variant} model")
    run = plan["runs"][variant]
    role = f"{variant}-agentic" if agentic else f"{variant}-responses"
    host_paths = [descriptor_path(args.plan, row, f"{variant} shard") for row in model["shards"]]
    runtime = inspect_runtime(run, args.output_dir, role, model_identity, host_paths)
    descriptor = plan["agentic_cases"] if agentic else plan["corpus"]
    cases, source_sha = load_rows(args.plan, descriptor, "agentic cases" if agentic else "quality corpus")
    rows = []
    for case in cases:
        case_id = case["id"]
        body = agentic_body(case) if agentic else response_body(case)
        request_desc = write_json(args.output_dir / role / f"{case_id}-request.json", body)
        result = request(run["endpoint"], body, args.timeout)
        response_desc = write_json(args.output_dir / role / f"{case_id}-response.json", result)
        if agentic:
            rows.append({"case_id": case_id, "variant": variant,
                "model_inventory_sha256": model_identity,
                "request_file": request_desc["path"], "request_sha256": request_desc["sha256"],
                "success": True, "errors": [],
                "response_file": response_desc["path"], "response_sha256": response_desc["sha256"]})
        else:
            rows.append({"case_id": case_id,
                "request_file": request_desc["path"], "request_sha256": request_desc["sha256"],
                "response_file": response_desc["path"], "response_sha256": response_desc["sha256"]})
    index_desc = write_jsonl(args.output_dir / f"{role}-index.jsonl", rows)
    record = {"schema": "ember.qwen3.8.quality-capture-phase.v1", "plan_sha256": plan_sha,
              "capture_driver_sha256": executing_driver_sha256(),
              "role": role, "source_sha256": source_sha, "model_inventory_sha256": model_identity,
              "runtime": runtime, "index": index_desc,
              "request_count": len(rows), "response_count": len(rows)}
    write_json(args.output_dir / f"{role}-run.json", record)


def candidate_side(case_id: str, candidate_sha: str) -> str:
    return "A" if int(hashlib.sha256(f"{case_id}\0{candidate_sha}".encode()).hexdigest(), 16) % 2 == 0 else "B"


def capture_judge(args: argparse.Namespace) -> None:
    plan, plan_sha = require_plan(args.plan)
    corpus, corpus_sha = load_rows(args.plan, plan["corpus"], "quality corpus")
    rubric_path = descriptor_path(args.plan, plan["rubric"], "rubric")
    rubric = json.loads(pinned(rubric_path, plan["rubric"]["sha256"], "rubric"))
    judge_path, judge_sha = artifact(plan["judge"]["artifact"], args.plan, "judge artifact")
    run = plan["runs"]["judge"]
    runtime = inspect_runtime(run, args.output_dir, "judge", judge_sha, [judge_path])
    settings = plan["judge"]["settings"]
    indexes: dict[str, dict[str, dict[str, str]]] = {}
    for variant in ("stock", "candidate"):
        phase = read_json(args.output_dir / f"{variant}-responses-run.json", f"{variant} response phase")
        if phase.get("plan_sha256") != plan_sha:
            raise CaptureError(f"{variant} response phase was produced from another plan")
        rows, _ = load_rows(args.output_dir, phase["index"], f"{variant} response index")
        indexes[variant] = {row["case_id"]: row for row in rows}
    verdicts = []
    for case in corpus:
        case_id = case["id"]
        captured: dict[str, tuple[str, str]] = {}
        for variant in ("stock", "candidate"):
            row = indexes[variant][case_id]
            response = json.loads(pinned(Path(row["response_file"]), row["response_sha256"], "response"))
            content = response["choices"][0]["message"].get("content") or ""
            captured[variant] = (row["response_sha256"], content)
        primary = candidate_side(case_id, captured["candidate"][0])
        for orientation in ("primary", "reverse"):
            cside = primary if orientation == "primary" else ("B" if primary == "A" else "A")
            sside = "B" if cside == "A" else "A"
            presented = {cside: captured["candidate"][0], sside: captured["stock"][0]}
            responses = {
                cside: {"sha256": captured["candidate"][0], "content": captured["candidate"][1]},
                sside: {"sha256": captured["stock"][0], "content": captured["stock"][1]},
            }
            evidence_request = {
                "schema_version": 1, "case_id": case_id, "orientation": orientation,
                "rubric_sha256": plan["rubric"]["sha256"], "system": rubric,
                "messages": case["messages"], "responses": responses,
                "tools": [SUBMIT_VERDICT_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "submit_verdict"}},
                "sampling": settings,
                "runtime": {"judge_artifact_sha256": judge_sha,
                            "image": runtime["image"], "engine_revision": runtime["engine_revision"]},
            }
            evidence_desc = write_json(args.output_dir / "judge" / f"{case_id}-{orientation}-request.json", evidence_request)
            api_body = {"model": "qwen3.8-quality-judge",
                "messages": [{"role": "system", "content": json.dumps(rubric, ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"))},
                    {"role": "user", "content": json.dumps({"messages": case["messages"],
                        "responses": responses}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}],
                "tools": [SUBMIT_VERDICT_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "submit_verdict"}},
                "max_tokens": 512, "temperature": 0, "seed": settings["seed"], "stream": False}
            api_desc = write_json(args.output_dir / "judge" / f"{case_id}-{orientation}-api-request.json", api_body)
            result = request(run["endpoint"], api_body, args.timeout)
            response_desc = write_json(args.output_dir / "judge" / f"{case_id}-{orientation}-response.json", result)
            verdicts.append({"case_id": case_id, "orientation": orientation, "presented": presented,
                "judge_artifact_sha256": judge_sha, "judge_image": runtime["image"],
                "rubric_sha256": plan["rubric"]["sha256"],
                "request_file": evidence_desc["path"], "request_sha256": evidence_desc["sha256"],
                "api_request_file": api_desc["path"], "api_request_sha256": api_desc["sha256"],
                "response_file": response_desc["path"], "response_sha256": response_desc["sha256"]})
    index_desc = write_jsonl(args.output_dir / "judge-index.jsonl", verdicts)
    write_json(args.output_dir / "judge-run.json", {
        "schema": "ember.qwen3.8.quality-capture-phase.v1", "plan_sha256": plan_sha,
        "capture_driver_sha256": executing_driver_sha256(),
        "role": "judge", "source_sha256": corpus_sha, "rubric_sha256": plan["rubric"]["sha256"],
        "runtime": runtime, "index": index_desc,
        "request_count": len(verdicts), "response_count": len(verdicts)})


def capture_driver_artifact(output: Path) -> dict[str, Any]:
    source = Path(__file__).resolve()
    target = output / "runtime" / "qwen_quality_capture.py"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
        with source.open("rb") as stream, os.fdopen(descriptor, "wb") as destination:
            shutil.copyfileobj(stream, destination)
            destination.flush()
            os.fsync(destination.fileno())
    digest, size = digest_file(target)
    source_digest, source_size = digest_file(source)
    if (digest, size) != (source_digest, source_size):
        raise CaptureError("captured driver differs from executing driver")
    return {"path": str(target), "sha256": digest, "bytes": size}


def assemble(args: argparse.Namespace) -> None:
    plan, plan_sha = require_plan(args.plan)
    phases = {name: read_json(args.output_dir / f"{name}-run.json", name)
              for name in ("stock-responses", "candidate-responses", "stock-agentic",
                           "candidate-agentic", "judge")}
    if any(row.get("plan_sha256") != plan_sha for row in phases.values()):
        raise CaptureError("capture phases do not share the immutable plan")
    driver = capture_driver_artifact(args.output_dir)
    if any(row.get("capture_driver_sha256") != driver["sha256"] for row in phases.values()):
        raise CaptureError("capture phases were not produced by this exact capture driver")
    stock_inv = inventory(plan["models"]["stock"], args.plan, "stock model")
    candidate_inv = inventory(plan["models"]["candidate"], args.plan, "candidate model")
    agentic_rows = []
    for variant in ("stock", "candidate"):
        rows, _ = load_rows(args.output_dir, phases[f"{variant}-agentic"]["index"], "agentic phase")
        agentic_rows.extend(rows)
    agentic_desc = write_jsonl(args.output_dir / "agentic-index.jsonl", agentic_rows)
    corpus_sha = plan["corpus"]["sha256"]
    agentic_sha = plan["agentic_cases"]["sha256"]
    runtime_attestation = {
        "schema": "ember.qwen3.8.quality-runtime-capture.v2",
        "release_scope": {"modality": "text_only", "multimodal_release_claim": False,
                          "vision_mmproj_differential_pass": False},
        "capture_driver": driver,
        "response_runs": {}, "judge_run": {}, "agentic_runs": {},
    }
    for variant, inv in (("stock", stock_inv), ("candidate", candidate_inv)):
        phase = phases[f"{variant}-responses"]
        runtime_attestation["response_runs"][variant] = {
            "runtime": phase["runtime"], "model_inventory_sha256": inv,
            "corpus_sha256": corpus_sha, "response_index_sha256": phase["index"]["sha256"],
            "request_count": phase["request_count"], "response_count": phase["response_count"]}
        phase = phases[f"{variant}-agentic"]
        runtime_attestation["agentic_runs"][variant] = {
            "runtime": phase["runtime"], "model_inventory_sha256": inv,
            "cases_sha256": agentic_sha, "results_index_sha256": agentic_desc["sha256"],
            "request_count": phase["request_count"], "response_count": phase["response_count"]}
    judge_sha = plan["judge"]["artifact"]["sha256"]
    judge_phase = phases["judge"]
    runtime_attestation["judge_run"] = {
        "runtime": judge_phase["runtime"], "judge_artifact_sha256": judge_sha,
        "rubric_sha256": plan["rubric"]["sha256"], "corpus_sha256": corpus_sha,
        "stock_response_index_sha256": phases["stock-responses"]["index"]["sha256"],
        "candidate_response_index_sha256": phases["candidate-responses"]["index"]["sha256"],
        "verdict_index_sha256": judge_phase["index"]["sha256"],
        "request_count": judge_phase["request_count"], "response_count": judge_phase["response_count"]}
    attestation_desc = write_json(args.output_dir / "runtime-attestation.json", runtime_attestation)
    contract = {"schema_version": 2, "contract_id": plan["contract_id"],
        "judge": {"artifact": plan["judge"]["artifact"],
                  "image": judge_phase["runtime"]["image"],
                  "engine_revision": judge_phase["runtime"]["engine_revision"],
                  "settings": plan["judge"]["settings"]},
        "rubric": plan["rubric"], "corpus": plan["corpus"], "models": plan["models"],
        "responses": {"stock": phases["stock-responses"]["index"],
                      "candidate": phases["candidate-responses"]["index"]},
        "verdicts": judge_phase["index"],
        "agentic": {"cases": plan["agentic_cases"], "results": agentic_desc,
                    "stock_variant": "stock", "candidate_variant": "candidate",
                    "model_inventories": {"stock": stock_inv, "candidate": candidate_inv}},
        "runtime_attestation": attestation_desc}
    result = write_json(args.output_dir / "quality-contract.json", contract)
    print(json.dumps(result, sort_keys=True))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("responses", "agentic"):
        child = sub.add_parser(name)
        child.add_argument("--variant", choices=("stock", "candidate"), required=True)
    sub.add_parser("judge")
    sub.add_parser("assemble")
    return parser.parse_args(list(argv))


def require_absolute_cli_paths(args: argparse.Namespace) -> None:
    """Keep every emitted descriptor independent of the caller's cwd."""
    if not args.plan.is_absolute():
        raise CaptureError("--plan must be an absolute path")
    if not args.output_dir.is_absolute():
        raise CaptureError("--output-dir must be an absolute path")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        require_absolute_cli_paths(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "responses":
            capture_variant(args, agentic=False)
        elif args.command == "agentic":
            capture_variant(args, agentic=True)
        elif args.command == "judge":
            capture_judge(args)
        else:
            assemble(args)
    except (CaptureError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"qwen-quality-capture: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
