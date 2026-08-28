#!/usr/bin/env python3
"""Execute one digest-bound runner-local Qwen quality descriptor request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import qwen_quality_descriptor as descriptor


REQUEST_SCHEMA = "ember.qwen3.8.quality-descriptor-request.v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class RequestError(ValueError):
    pass


def fail(message: str) -> None:
    raise RequestError(message)


def pinned(value: Any, label: str) -> tuple[Path, str]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or HEX64.fullmatch(str(value.get("sha256", ""))) is None):
        fail(f"{label} descriptor is malformed")
    return Path(value["path"]), value["sha256"]


def request_args(request_path: Path, request_sha256: str,
                 ember_revision: str) -> argparse.Namespace:
    value, _ = descriptor.exact_json(
        request_path, request_sha256, "quality descriptor request")
    expected = {
        "schema", "ember_revision", "phase", "phase_plan", "stock_build_record",
        "candidate_build_record", "candidate_id", "judge_inventory",
        "model_runtime_image", "judge_runtime_image", "quality_output_root",
        "capture_plan_output", "phase_descriptor_output", "publishes", "deletes",
    }
    if (set(value) != expected or value.get("schema") != REQUEST_SCHEMA
            or value.get("publishes") is not False
            or value.get("deletes") is not False):
        fail("quality descriptor request schema/lifecycle differs")
    if (HEX40.fullmatch(ember_revision) is None
            or value.get("ember_revision") != ember_revision):
        fail("quality descriptor request Ember revision differs")
    if value.get("phase") not in {"sweep", "final"}:
        fail("quality descriptor request phase differs")
    phase_plan, phase_plan_sha = pinned(value.get("phase_plan"), "phase plan")
    stock_record, stock_record_sha = pinned(
        value.get("stock_build_record"), "stock build record")
    candidate_record, candidate_record_sha = pinned(
        value.get("candidate_build_record"), "candidate build record")
    judge_inventory, judge_inventory_sha = pinned(
        value.get("judge_inventory"), "judge inventory")
    candidate_id = value.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        fail("quality descriptor request candidate id is malformed")

    def absolute(name: str) -> Path:
        raw = value.get(name)
        if (not isinstance(raw, str) or not Path(raw).is_absolute()
                or Path(raw) == Path("/")):
            fail(f"quality descriptor request {name} is malformed")
        return Path(raw)

    return argparse.Namespace(
        phase=value["phase"], phase_plan=phase_plan,
        phase_plan_sha256=phase_plan_sha, stock_build_record=stock_record,
        stock_build_record_sha256=stock_record_sha,
        candidate_build_record=candidate_record,
        candidate_build_record_sha256=candidate_record_sha,
        candidate_id=candidate_id, judge_inventory=judge_inventory,
        judge_inventory_sha256=judge_inventory_sha,
        ember_revision=ember_revision,
        model_runtime_image=value.get("model_runtime_image"),
        judge_runtime_image=value.get("judge_runtime_image"),
        quality_output_root=absolute("quality_output_root"),
        capture_plan_output=absolute("capture_plan_output"),
        output=absolute("phase_descriptor_output"),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--request", type=Path, required=True)
    result.add_argument("--request-sha256", required=True)
    result.add_argument("--ember-revision", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        generated = request_args(
            args.request.absolute(), args.request_sha256, args.ember_revision)
        capture, phase = descriptor.generate(generated)
    except (RequestError, descriptor.DescriptorError, OSError, ValueError) as exc:
        print(f"qwen-quality-request: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "capture_plan": {
            "path": str(generated.capture_plan_output),
            "sha256": descriptor.sha256_file(generated.capture_plan_output),
        },
        "phase_descriptor": {
            "path": str(generated.output),
            "sha256": descriptor.sha256_file(generated.output),
        },
        "contract_id": capture["contract_id"],
        "phase": phase["phase"],
        "quality_output_root": phase["output_dir"],
        "request": {"path": str(args.request.absolute()),
                    "sha256": args.request_sha256},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
