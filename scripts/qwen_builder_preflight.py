#!/usr/bin/env python3
"""Report whether a runner can build the pinned Qwen3.8 release artifacts.

This is intentionally read-only.  It checks the small set of documented model
and Hugging Face cache locations, never walks arbitrary mounts, and prints no
environment values or credentials.  The quantization workflow consumes the
JSON record after a feature-branch dispatch to decide whether the runner can
use the ordinary BF16 conversion or needs a lower-memory streaming path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil


SOURCE_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
MINIMUM_RAM_GIB = 256
MINIMUM_DISK_GIB = 1024


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            fields = raw.split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def disk_record(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        usage = shutil.disk_usage(path)
        result.update({
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / 1024**3, 2),
        })
    except OSError as exc:
        result["error"] = type(exc).__name__
    return result


def candidate_snapshots() -> list[Path]:
    roots = [
        Path("/models/Qwen3.8-Flash-Next"),
        Path("/srv/models/Qwen3.8-Flash-Next"),
        Path("/scratch/Qwen3.8-Flash-Next"),
    ]
    cache_roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home and Path(hf_home).is_absolute():
        cache_roots.append(Path(hf_home))
    cache_roots.append(Path.home() / ".cache/huggingface")
    for root in cache_roots:
        roots.append(
            root / "hub/models--Qwen--Qwen3.8-Flash-Next/snapshots" /
            SOURCE_REVISION
        )
    candidates: list[Path] = []
    for root in roots:
        candidates.extend((root / "snapshots" / SOURCE_REVISION, root))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def snapshot_record(path: Path) -> dict[str, object]:
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    return {
        "path": str(path),
        "config_present": config.is_file(),
        "index_present": index.is_file(),
        "complete_metadata": config.is_file() and index.is_file(),
    }


def build_record() -> dict[str, object]:
    memory = meminfo()
    ram = memory.get("MemTotal", 0)
    swap = memory.get("SwapTotal", 0)
    disks = [disk_record(Path(path)) for path in (".", "/scratch", "/models", "/srv/models")]
    maximum_free = max(
        (int(item.get("free_bytes", 0)) for item in disks), default=0
    )
    snapshots = [
        record for record in map(snapshot_record, candidate_snapshots())
        if record["config_present"] or record["index_present"]
    ]
    return {
        "schema": "ember.qwen3.8.builder-preflight.v1",
        "source_revision": SOURCE_REVISION,
        "host": {
            "node": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
        },
        "memory": {
            "ram_bytes": ram,
            "ram_gib": round(ram / 1024**3, 2),
            "swap_bytes": swap,
            "swap_gib": round(swap / 1024**3, 2),
            "ordinary_conversion_minimum_gib": MINIMUM_RAM_GIB,
            "ordinary_conversion_ram_fit": ram >= MINIMUM_RAM_GIB * 1024**3,
        },
        "filesystems": disks,
        "ordinary_conversion_disk_fit": maximum_free >= MINIMUM_DISK_GIB * 1024**3,
        "ordinary_conversion_minimum_free_gib": MINIMUM_DISK_GIB,
        "devices": {
            "kfd_readable": os.access("/dev/kfd", os.R_OK),
            "dri_present": Path("/dev/dri").is_dir(),
        },
        "tools": {
            name: shutil.which(name) is not None
            for name in ("docker", "git", "cmake", "python3")
        },
        "pinned_snapshot_candidates": snapshots,
        "pinned_snapshot_metadata_found": any(
            bool(item["complete_metadata"]) for item in snapshots
        ),
        "mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = build_record()
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
