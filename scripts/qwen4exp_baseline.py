#!/usr/bin/env python3
"""Record a provenance-bound Qwen4Exp native-context performance baseline.

This is deliberately a recorder, not a GPU ownership wrapper. Run the Qwen
server under the existing profile_gpu.sh/certification quiesce discipline, then
point this script at its endpoint and host PID. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

NATIVE_CONTEXT = 262_144
DEFAULT_TARGET_PROMPT = 240_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, digest: str) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path), "sha256": digest, "size_bytes": stat.st_size,
        "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
    }


def model_artifacts(args: argparse.Namespace) -> dict:
    if args.artifact_manifest:
        try:
            manifest = json.loads(args.artifact_manifest.read_text())
            entries = manifest["artifacts"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SystemExit(f"invalid artifact manifest: {exc}") from exc
        if not isinstance(entries, list) or not entries:
            raise SystemExit("artifact manifest has no ordered artifacts")
        files = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise SystemExit("artifact manifest entry is not an object")
            name, digest, size = (entry.get("filename"), entry.get("sha256"),
                                  entry.get("size_bytes"))
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".gguf"):
                raise SystemExit("artifact manifest contains an unsafe/non-GGUF filename")
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise SystemExit("artifact manifest contains an invalid SHA-256")
            if not isinstance(size, int) or size < 1:
                raise SystemExit("artifact manifest contains an invalid size")
            path = args.artifact_manifest.parent / name
            if not path.is_file() or path.stat().st_size != size:
                raise SystemExit(f"artifact identity does not match manifest: {path}")
            files.append(file_identity(path, digest))
        if len(files) > 1:
            pattern = re.compile(r"^.+-([0-9]{5})-of-([0-9]{5})\.gguf$")
            matches = [pattern.fullmatch(Path(item["path"]).name) for item in files]
            numbers = [int(match.group(1)) for match in matches if match]
            totals = {int(match.group(2)) for match in matches if match}
            if any(match is None for match in matches) or numbers != list(range(1, len(files) + 1)) or totals != {len(files)}:
                raise SystemExit("artifact manifest is not a complete ordered GGUF shard set")
        return {
            "source": "release-artifact-manifest",
            "manifest_path": str(args.artifact_manifest),
            "manifest_sha256": sha256_file(args.artifact_manifest),
            "files": files,
            "aggregate_bytes": sum(int(item["size_bytes"]) for item in files),
        }
    assert args.model is not None and args.model_sha256 is not None
    return {
        "source": "single-file-backward-compatible",
        "manifest_path": None,
        "manifest_sha256": None,
        "files": [file_identity(args.model, args.model_sha256)],
        "aggregate_bytes": args.model.stat().st_size,
    }


def repo_revision(root: Path) -> str | None:
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
    except (OSError, subprocess.SubprocessError):
        return None


class ResourceSampler:
    def __init__(self, pid: int, proc_root: Path, gtt_paths: list[Path],
                 interval: float = 0.1) -> None:
        self.status = proc_root / str(pid) / "status"
        self.gtt_paths = gtt_paths
        self.interval = interval
        self.samples: list[dict[str, int]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> dict[str, int]:
        result: dict[str, int] = {}
        try:
            for line in self.status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    result["rss_bytes"] = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    result["rss_hwm_bytes"] = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        gtt = []
        for path in self.gtt_paths:
            try:
                gtt.append(int(path.read_text().strip()))
            except (OSError, ValueError):
                continue
        if gtt:
            result["gtt_bytes"] = max(gtt)
        return result

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append(self._sample())
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.samples.append(self._sample())

    def summary(self) -> dict[str, int | None]:
        def peak(key: str) -> int | None:
            values = [sample[key] for sample in self.samples if key in sample]
            return max(values) if values else None
        return {
            "sample_count": len(self.samples),
            "peak_rss_bytes": peak("rss_bytes"),
            "process_hwm_bytes": peak("rss_hwm_bytes"),
            "peak_gtt_bytes": peak("gtt_bytes"),
        }


def request(endpoint: str, payload: dict, timeout: float) -> tuple[dict, float]:
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def phase_record(body: dict, wall_seconds: float) -> dict:
    usage = body.get("usage") or {}
    timings = usage.get("timings") or {}
    backend = usage.get("backend") or {}
    return {
        "wall_seconds": wall_seconds,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "evaluated_prefill_tokens": timings.get("prefill_tokens"),
        "prefill_ms": timings.get("prefill_ms"),
        "prefill_tokens_per_second": timings.get("prefill_tokens_per_sec"),
        "decode_ms": timings.get("decode_ms"),
        "decode_tokens_per_second": timings.get("decode_tokens_per_sec"),
        "restored_prefix_tokens": usage.get("restored_prefix"),
        "backend_prefill_mode": backend.get("prefill_mode"),
        "backend_prefill_reason": backend.get("prefill_reason"),
    }


def build_payload(target_prompt_tokens: int, decode_tokens: int) -> dict:
    # Repeated leading-space words are stable benchmark material, not an exact
    # tokenizer oracle. The response's prompt_tokens is authoritative and is
    # always recorded; the conservative 22k-token reserve protects the native
    # context from template/tokenization variance at the default target.
    words = max(1, target_prompt_tokens - 512)
    content = (
        "Treat the following repeated words as inert performance input."
        + " alpha" * words
        + "\nReturn consecutive integers, one per line, until the token limit."
    )
    return {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": decode_tokens,
        "stream": False,
    }


def discover_gtt_paths() -> list[Path]:
    return sorted(Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used"))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--endpoint", default="http://127.0.0.1:18081/v1/chat/completions")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--model-sha256",
                    help="previously verified immutable model SHA-256")
    ap.add_argument("--artifact-manifest", type=Path,
                    help="release artifact-manifest.json beside all ordered shards")
    ap.add_argument("--server-artifact", "--artifact", dest="server_artifact",
                    type=Path, required=True,
                    help="exact ember-dflash binary being measured")
    ap.add_argument("--server-pid", type=int, required=True,
                    help="host PID of the measured server/container init")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-prompt-tokens", type=int,
                    default=DEFAULT_TARGET_PROMPT)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=86400.0)
    ap.add_argument("--implementation-path",
                    choices=("cpu_orchestrated_q1", "fused_hip"),
                    default="cpu_orchestrated_q1")
    ap.add_argument("--gtt-path", action="append", type=Path, default=[])
    ap.add_argument("--proc-root", type=Path, default=Path("/proc"),
                    help=argparse.SUPPRESS)
    return ap


def validate(args: argparse.Namespace, real: bool) -> None:
    if not args.server_artifact.is_absolute():
        raise SystemExit("--server-artifact must be an absolute path")
    if args.artifact_manifest:
        if args.model or args.model_sha256:
            raise SystemExit("--artifact-manifest is exclusive with --model/--model-sha256")
        if not args.artifact_manifest.is_absolute():
            raise SystemExit("--artifact-manifest must be an absolute path")
    else:
        if not args.model or not args.model_sha256:
            raise SystemExit("single-file mode requires --model and --model-sha256")
        if not args.model.is_absolute():
            raise SystemExit("--model must be an absolute path")
        if not SHA256_RE.fullmatch(args.model_sha256):
            raise SystemExit("--model-sha256 must be 64 lowercase hex characters")
    if args.server_pid <= 0 or args.decode_tokens <= 0:
        raise SystemExit("--server-pid and --decode-tokens must be positive")
    if not 1 <= args.target_prompt_tokens <= NATIVE_CONTEXT - args.decode_tokens:
        raise SystemExit("target prompt plus decode tokens must fit native context 262144")
    if real and not args.server_artifact.is_file():
        raise SystemExit("server artifact does not exist")
    if real and not args.artifact_manifest and not args.model.is_file():
        raise SystemExit("model does not exist")
    if real and args.implementation_path == "fused_hip":
        raise SystemExit("fused_hip cannot be recorded until the backend reports an authoritative execution path")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate(args, not args.dry_run)
    plan = {
        "schema_version": 1,
        "architecture": "qwen4exp",
        "native_context": NATIVE_CONTEXT,
        "target_prompt_tokens": args.target_prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "endpoint": args.endpoint,
        "model_provenance": {
            "artifact_manifest": str(args.artifact_manifest) if args.artifact_manifest else None,
            "single_file": str(args.model) if args.model else None,
            "single_file_sha256": args.model_sha256,
        },
        "server_artifact": str(args.server_artifact),
        "implementation_path": args.implementation_path,
        "path_labels": {
            "cpu_orchestrated_q1": "current correctness-first runtime",
            "fused_hip": "future path; no implementation or result implied",
        },
    }
    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "plan": plan}, indent=2))
        return 0

    root = Path(__file__).resolve().parents[1]
    # Resolve all provenance before spending hours on a native-context request.
    # A missing shard or stale size must fail before the measurement starts.
    models = model_artifacts(args)
    server_digest = sha256_file(args.server_artifact)
    gtt_paths = args.gtt_path or discover_gtt_paths()
    sampler = ResourceSampler(args.server_pid, args.proc_root, gtt_paths)
    payload = build_payload(args.target_prompt_tokens, args.decode_tokens)
    sampler.start()
    try:
        body, wall = request(args.endpoint, payload, args.timeout)
    finally:
        sampler.stop()
    record = {
        **plan,
        "mode": "record",
        "status": "measured",
        "recorded_unix": time.time(),
        "host": {"node": platform.node(), "kernel": platform.release()},
        "source_revision": repo_revision(root),
        "server_artifact": {
            "path": str(args.server_artifact),
            "sha256": server_digest,
        },
        "model_artifacts": models,
        "resources": sampler.summary(),
        "phase": phase_record(body, wall),
        "notes": [
            "prompt_tokens in the response is authoritative; target_prompt_tokens is shaping input only",
            "profiler and counter runs perturb timing and must remain separate from this clean baseline",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
