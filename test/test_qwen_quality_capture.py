#!/usr/bin/env python3
"""GPU-free fixtures for the live Qwen quality capture driver."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen_quality_capture", ROOT / "scripts" / "qwen_quality_capture.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def encoded(value: Any) -> bytes:
    return MODULE.canonical(value)


def descriptor(path: Path, value: bytes, *, sized: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    result: dict[str, Any] = {"path": str(path), "sha256": MODULE.digest_bytes(value)}
    if sized:
        result["bytes"] = len(value)
    return result


def main() -> int:
    try:
        MODULE.require_absolute_cli_paths(argparse.Namespace(
            plan=Path("relative-plan.json"), output_dir=Path("/absolute-output")))
    except MODULE.CaptureError as exc:
        assert "--plan" in str(exc)
    else:
        raise AssertionError("relative quality-capture plan was accepted")
    try:
        MODULE.require_absolute_cli_paths(argparse.Namespace(
            plan=Path("/absolute-plan.json"), output_dir=Path("relative-output")))
    except MODULE.CaptureError as exc:
        assert "--output-dir" in str(exc)
    else:
        raise AssertionError("relative quality-capture output was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_file = root / "stock.gguf"
        model_row = descriptor(model_file, b"real model bytes", sized=True)
        model = {
            "candidate_id": "stock", "build_record_sha256": "1" * 64,
            "intervention_manifest_sha256": "2" * 64, "profile_sha256": "3" * 64,
            "quantization_overrides_sha256": "4" * 64, "shards": [model_row],
        }
        case = {"id": "one", "suite": "coherence",
                "messages": [{"role": "user", "content": "answer"}],
                "expected": {"max_tokens": 16}, "request_sha256": "5" * 64}
        corpus = descriptor(root / "corpus.jsonl", encoded(case))
        rubric = descriptor(root / "rubric.json", encoded({"rubric": "fixture"}))
        agentic = descriptor(root / "agentic.jsonl", encoded({
            "id": "agent", "messages": case["messages"], "max_tokens": 8,
        }))
        judge_artifact = descriptor(root / "judge.gguf", b"judge", sized=True)
        image = "ghcr.io/otheru-ai/ember@sha256:" + "a" * 64
        run = {"container": "fixture", "endpoint": "http://127.0.0.1:18080/v1/chat/completions",
               "image": image, "loaded_path": "/models/stock.gguf"}
        plan = {"schema": "ember.qwen3.8.quality-capture-plan.v1", "contract_id": "fixture",
                "corpus": corpus, "rubric": rubric, "agentic_cases": agentic,
                "models": {"stock": model, "candidate": {**model, "candidate_id": "candidate"}},
                "judge": {"artifact": judge_artifact, "settings": {
                    "temperature": 0, "seed": 7301, "batch_size": 1, "target_only": True,
                    "speculative_decode": False, "required_tool": "submit_verdict"}},
                "runs": {"stock": run, "candidate": run, "judge": run}}
        plan_path = root / "plan.json"
        plan_path.write_bytes(encoded(plan))
        binary = root / "source-binary"
        binary.write_bytes(b"live container binary")
        environment = ["DFLASH_DS4_SPEC=0"]

        def fake_command(values: list[str]) -> str:
            if values[:3] == ["docker", "container", "inspect"]:
                return json.dumps([{
                    "Id": "b" * 64, "Image": "sha256:" + "c" * 64,
                    "State": {"Running": True}, "Config": {"Env": environment},
                    "Mounts": [{"Source": str(root), "Destination": "/models", "RW": False}],
                    "Path": "/usr/local/bin/ember-dflash",
                    "Args": ["-m", "/models/stock.gguf"],
                }])
            if values[:3] == ["docker", "image", "inspect"]:
                return json.dumps([{
                    "Id": "sha256:" + "c" * 64, "RepoDigests": [image],
                    "Config": {"Labels": {"org.opencontainers.image.revision": "d" * 40}},
                }])
            if values[:3] == ["docker", "exec", "fixture"]:
                if values[-1] in MODULE.TARGET_ENV:
                    return (environment[0].split("=", 1)[1] + "\n"
                            if values[-1] == "DFLASH_DS4_SPEC" else "\n")
                return "  Name: gfx1151\n"
            if values[:2] == ["docker", "cp"]:
                shutil.copyfile(binary, values[-1])
                return ""
            raise AssertionError(values)

        posted: list[dict[str, Any]] = []

        def fake_request(endpoint: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
            assert endpoint == run["endpoint"] and timeout == 12
            posted.append(body)
            return {"choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}

        old_command, old_request = MODULE.command, MODULE.request
        MODULE.command, MODULE.request = fake_command, fake_request
        try:
            output = root / "capture"
            MODULE.capture_variant(argparse.Namespace(
                plan=plan_path, output_dir=output, timeout=12, variant="stock"), agentic=False)
            phase = json.loads((output / "stock-responses-run.json").read_text())
            assert phase["request_count"] == phase["response_count"] == 1
            assert phase["runtime"]["image_id"] == "sha256:" + "c" * 64
            assert phase["runtime"]["inspection"]["sha256"] == MODULE.digest_file(
                output / "runtime/stock-responses-inspection.json")[0]
            assert posted == [{
                "model": "qwen3.8-flash-next", "messages": case["messages"],
                "max_tokens": 16, "temperature": 0, "seed": 7301, "stream": False,
            }]
            index = json.loads((output / "stock-responses-index.jsonl").read_text())
            captured_body = json.loads(Path(index["request_file"]).read_text())
            assert captured_body == posted[0]

            environment[0] = "DFLASH_DS4_SPEC=1"
            try:
                MODULE.inspect_runtime(run, root / "rejected", "stock-responses", "e" * 64,
                                       [model_file])
            except MODULE.CaptureError as exc:
                assert "target-only" in str(exc)
            else:
                raise AssertionError("speculative runtime was accepted as target-only evidence")
        finally:
            MODULE.command, MODULE.request = old_command, old_request
    print("qwen quality capture: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
