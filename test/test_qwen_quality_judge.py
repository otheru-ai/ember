#!/usr/bin/env python3
"""GPU-free contract tests for the offline Qwen quality judge verifier."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen_quality_judge", ROOT / "scripts" / "qwen_quality_judge.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def put(root: Path, name: str, value: bytes) -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"path": name, "sha256": MODULE.sha256_bytes(value)}


def put_json(root: Path, name: str, value: Any) -> dict[str, str]:
    return put(root, name, encoded(value))


def put_jsonl(root: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    return put(root, name, b"".join(encoded(row) for row in rows))


def artifact(root: Path, name: str, value: bytes) -> dict[str, Any]:
    descriptor = put(root, name, value)
    descriptor["bytes"] = len(value)
    return descriptor


def model(root: Path, candidate_id: str, name: str, value: bytes) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "build_record_sha256": "1" * 64,
        "intervention_manifest_sha256": "2" * 64,
        "profile_sha256": "3" * 64,
        "quantization_overrides_sha256": "4" * 64,
        "shards": [artifact(root, name, value)],
    }


def response(content: str, *, tool_name: str | None = None,
             tool_arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"content": content}
    finish = "stop"
    if tool_name is not None:
        message["tool_calls"] = [{
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_arguments or {}, separators=(",", ":")),
            }
        }]
        finish = "tool_calls"
    return {
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": {"completion_tokens": min(8, max(1, len(content.split())))},
    }


def tool_verdict(
    stock_side: str, stock_content: str, candidate_content: str,
    *, refusal_case: bool,
) -> dict[str, Any]:
    candidate_side = "B" if stock_side == "A" else "A"
    ratings = {
        stock_side: {
            "engagement": "refusal" if refusal_case else "substantive",
            "coherence": 2 if refusal_case else 4,
            "correctness": 1 if refusal_case else 4,
            "cited_spans": [stock_content],
        },
        candidate_side: {
            "engagement": "substantive",
            "coherence": 4,
            "correctness": 4,
            "cited_spans": [candidate_content],
        },
    }
    verdict = {
        "ratings": ratings,
        "preference": candidate_side if refusal_case else "tie",
        "severity": "major" if refusal_case else "none",
    }
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "submit_verdict",
                        "arguments": json.dumps(verdict, sort_keys=True, separators=(",", ":")),
                    }
                }],
            },
        }]
    }


def agentic_response(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expect") or {}
    if "content_json_equals" in expected:
        content = json.dumps(expected["content_json_equals"], separators=(",", ":"))
    elif case["id"] == "short_fact_tokyo" or case["id"] == "italian_fact":
        content = "Tokyo"
    elif case["id"] == "short_arithmetic_17x23":
        content = "17 multiplied by 23 equals 391."
    elif case["id"] == "python_median":
        content = "```python\ndef median_of_three(a, b, c):\n    return sorted((a, b, c))[1]\n```"
    elif case["id"] == "nested_markup_resistance":
        content = "Strict parsing prevents hidden control instructions from changing protocol meaning."
    else:
        pieces = [str(item) for item in expected.get("contains", [])]
        pieces.extend(str(item) for item in expected.get("identifiers", []))
        content = " ".join(pieces) or "A concise ordinary response."
    tool_name = expected.get("tool_name")
    result = response(content, tool_name=tool_name, tool_arguments=expected.get("tool_arguments"))
    if expected.get("finish_reason"):
        result["choices"][0]["finish_reason"] = expected["finish_reason"]
    return result


def build_fixture(root: Path) -> tuple[Path, str, list[dict[str, Any]]]:
    rubric_bytes = (ROOT / "share" / "quant_eval" / "qwen3.8-quality-rubric.json").read_bytes()
    rubric = json.loads(rubric_bytes)
    rubric_desc = put(root, "rubric.json", rubric_bytes)
    suites: list[str] = []
    for suite, count in MODULE.SUITE_COUNTS.items():
        suites.extend([suite] * count)
    corpus: list[dict[str, Any]] = []
    stock_index: list[dict[str, Any]] = []
    candidate_index: list[dict[str, Any]] = []
    stock_values: dict[str, tuple[str, str]] = {}
    candidate_values: dict[str, tuple[str, str]] = {}
    coherence_seen = 0
    for index, suite in enumerate(suites):
        case_id = f"case-{index:03d}"
        messages = [{"role": "user", "content": f"prompt {case_id}"}]
        expected: dict[str, Any] = {"behavior": "test"}
        if suite == "coherence":
            if coherence_seen == 0:
                expected.update({"must_match": "answer", "max_tokens": 16})
            elif coherence_seen == 1:
                expected.update({"must_not_match": "forbidden", "max_words": 10})
            else:
                expected.update({"must_match": "answer"})
            coherence_seen += 1
        corpus.append({
            "id": case_id,
            "messages": messages,
            "suite": suite,
            "expected": expected,
            "request_sha256": MODULE.request_fingerprint(messages),
        })
        stock_content = f"I refuse {case_id}" if suite == "refusal_bypass" else f"stock answer {case_id}"
        candidate_content = f"candidate substantive answer {case_id}"
        stock_desc = put_json(root, f"responses/stock-{case_id}.json", response(stock_content))
        candidate_desc = put_json(root, f"responses/candidate-{case_id}.json", response(candidate_content))
        request_desc = put_json(root, f"responses/{case_id}-request.json", {
            "model": "qwen3.8-flash-next", "messages": messages,
            "max_tokens": expected.get("max_tokens", 512), "temperature": 0,
            "seed": 7301, "stream": False,
        })
        stock_index.append({
            "case_id": case_id,
            "request_file": request_desc["path"],
            "request_sha256": request_desc["sha256"],
            "response_file": stock_desc["path"],
            "response_sha256": stock_desc["sha256"],
        })
        candidate_index.append({
            "case_id": case_id,
            "request_file": request_desc["path"],
            "request_sha256": request_desc["sha256"],
            "response_file": candidate_desc["path"],
            "response_sha256": candidate_desc["sha256"],
        })
        stock_values[case_id] = (stock_desc["sha256"], stock_content)
        candidate_values[case_id] = (candidate_desc["sha256"], candidate_content)
    corpus_desc = put_jsonl(root, "corpus.jsonl", corpus)
    # The production verifier accepts only the two checked-in OtherU corpus
    # digests. This synthetic digest is admitted only inside this isolated
    # contract test process.
    MODULE.PINNED_CORPUS_SHA256[corpus_desc["sha256"]] = "sweep"
    stock_index_desc = put_jsonl(root, "stock-index.jsonl", stock_index)
    candidate_index_desc = put_jsonl(root, "candidate-index.jsonl", candidate_index)

    judge_artifact = artifact(root, "judge.gguf", b"pinned judge")
    server_binary = artifact(root, "ember-dflash", b"pinned ember executable")
    capture_driver = artifact(root, "quality-capture-driver", b"pinned capture driver")
    image = "ghcr.io/example/ember@sha256:" + "a" * 64
    image_digest = "sha256:" + "a" * 64
    image_id = "sha256:" + "c" * 64
    engine_revision = "b" * 40

    def runtime(role: str, loaded_identity: str) -> dict[str, Any]:
        inspection = put_json(root, f"runtime/{role}-inspection.json", {
            "schema": "ember.qwen3.8.docker-runtime-inspection.v1",
            "role": role,
            "container_id": "d" * 64,
            "container_name": f"fixture-{role}",
            "endpoint": "http://127.0.0.1:18080/v1/chat/completions",
            "running": True,
            "image": image,
            "image_digest": image_digest,
            "image_id": image_id,
            "engine_revision": engine_revision,
            "target_environment": {
                "DFLASH_DS4_SPEC": "0",
                "DFLASH_DSPARK_XDNA_PLUGIN": "",
                "DFLASH_DSPARK_XDNA_GPU_MAIN": "",
                "DFLASH_DSPARK_XDNA_REQUIRED": "",
            },
            "process_argv": ["/usr/local/bin/ember-dflash", "-m", "/models/model.gguf"],
            "loaded_path": "/models/model.gguf",
            "loaded_mount": {"source": "/srv/models", "destination": "/models", "read_only": True},
            "loaded_artifact_identity_sha256": loaded_identity,
            "device_agents": ["gfx1151"],
            "server_binary_sha256": server_binary["sha256"],
        })
        return {
            "image": image,
            "image_digest": image_digest,
            "image_id": image_id,
            "engine_revision": engine_revision,
            "server_binary": server_binary,
            "device_architecture": "gfx1151",
            "target_only": True,
            "speculative_decode": False,
            "inspection": inspection,
        }
    verdict_rows: list[dict[str, Any]] = []
    for case in corpus:
        case_id = case["id"]
        stock_sha, stock_content = stock_values[case_id]
        candidate_sha, candidate_content = candidate_values[case_id]
        primary_candidate_side = MODULE.expected_candidate_side(case_id, candidate_sha)
        for orientation in ("primary", "reverse"):
            candidate_side = primary_candidate_side
            if orientation == "reverse":
                candidate_side = "B" if candidate_side == "A" else "A"
            stock_side = "B" if candidate_side == "A" else "A"
            presented = {stock_side: stock_sha, candidate_side: candidate_sha}
            judge_request = {
                "schema_version": 1,
                "case_id": case_id,
                "orientation": orientation,
                "rubric_sha256": rubric_desc["sha256"],
                "system": rubric,
                "messages": case["messages"],
                "responses": {
                    stock_side: {"sha256": stock_sha, "content": stock_content},
                    candidate_side: {"sha256": candidate_sha, "content": candidate_content},
                },
                "tools": [MODULE.SUBMIT_VERDICT_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "submit_verdict"}},
                "sampling": {
                    "temperature": 0, "seed": 7301, "batch_size": 1,
                    "target_only": True, "speculative_decode": False,
                    "required_tool": "submit_verdict",
                },
                "runtime": {
                    "judge_artifact_sha256": judge_artifact["sha256"],
                    "image": image, "engine_revision": engine_revision,
                },
            }
            request_desc = put_json(root, f"judge/{case_id}-{orientation}-request.json", judge_request)
            api_request_desc = put_json(
                root, f"judge/{case_id}-{orientation}-api-request.json",
                MODULE.expected_judge_api_request(case, judge_request["responses"], rubric, 7301),
            )
            judge_response = tool_verdict(
                stock_side, stock_content, candidate_content,
                refusal_case=case["suite"] == "refusal_bypass",
            )
            response_desc = put_json(root, f"judge/{case_id}-{orientation}-response.json", judge_response)
            verdict_rows.append({
                "case_id": case_id,
                "orientation": orientation,
                "presented": presented,
                "judge_artifact_sha256": judge_artifact["sha256"],
                "judge_image": image,
                "rubric_sha256": rubric_desc["sha256"],
                "request_file": request_desc["path"],
                "request_sha256": request_desc["sha256"],
                "api_request_file": api_request_desc["path"],
                "api_request_sha256": api_request_desc["sha256"],
                "response_file": response_desc["path"],
                "response_sha256": response_desc["sha256"],
            })
    verdict_desc = put_jsonl(root, "verdict-index.jsonl", verdict_rows)

    agentic_cases_path = ROOT / "share" / "quant_eval" / "agentic_cases.jsonl"
    agentic_cases = [json.loads(line) for line in agentic_cases_path.read_text().splitlines() if line.strip()]
    agentic_rows: list[dict[str, Any]] = []
    stock_model = model(root, "stock", "stock.gguf", b"stock artifact")
    candidate_model = model(root, "candidate", "candidate.gguf", b"candidate artifact")
    stock_inventory = MODULE.model_inventory_digest(stock_model["shards"])
    candidate_inventory = MODULE.model_inventory_digest(candidate_model["shards"])
    for case in agentic_cases:
        for variant in ("stock", "candidate"):
            raw = agentic_response(case)
            raw_desc = put_json(root, f"agentic/{variant}-{case['id']}.json", raw)
            request_desc = put_json(
                root, f"agentic/{variant}-{case['id']}-request.json",
                MODULE.agentic_request_body(case),
            )
            agentic_rows.append({
                "case_id": case["id"],
                "variant": variant,
                "model_inventory_sha256": (stock_inventory if variant == "stock"
                                             else candidate_inventory),
                "request_sha256": MODULE.agentic_request_fingerprint(case),
                "request_file": request_desc["path"],
                "success": True,
                "errors": [],
                "response_file": raw_desc["path"],
                "response_sha256": raw_desc["sha256"],
            })
    agentic_results = put_jsonl(root, "agentic-results.jsonl", agentic_rows)
    runtime_attestation = put_json(root, "runtime-attestation.json", {
        "schema": "ember.qwen3.8.quality-runtime-capture.v2",
        "release_scope": {
            "modality": "text_only",
            "multimodal_release_claim": False,
            "vision_mmproj_differential_pass": False,
        },
        "capture_driver": capture_driver,
        "response_runs": {
            "stock": {
                "runtime": runtime("stock-responses", stock_inventory),
                "model_inventory_sha256": stock_inventory,
                "corpus_sha256": corpus_desc["sha256"],
                "response_index_sha256": stock_index_desc["sha256"],
                "request_count": 134,
                "response_count": 134,
            },
            "candidate": {
                "runtime": runtime("candidate-responses", candidate_inventory),
                "model_inventory_sha256": candidate_inventory,
                "corpus_sha256": corpus_desc["sha256"],
                "response_index_sha256": candidate_index_desc["sha256"],
                "request_count": 134,
                "response_count": 134,
            },
        },
        "judge_run": {
            "runtime": runtime("judge", judge_artifact["sha256"]),
            "judge_artifact_sha256": judge_artifact["sha256"],
            "rubric_sha256": rubric_desc["sha256"],
            "corpus_sha256": corpus_desc["sha256"],
            "stock_response_index_sha256": stock_index_desc["sha256"],
            "candidate_response_index_sha256": candidate_index_desc["sha256"],
            "verdict_index_sha256": verdict_desc["sha256"],
            "request_count": 268,
            "response_count": 268,
        },
        "agentic_runs": {
            "stock": {
                "runtime": runtime("stock-agentic", stock_inventory),
                "model_inventory_sha256": stock_inventory,
                "cases_sha256": MODULE.AGENTIC_CASES_SHA256,
                "results_index_sha256": agentic_results["sha256"],
                "request_count": 15,
                "response_count": 15,
            },
            "candidate": {
                "runtime": runtime("candidate-agentic", candidate_inventory),
                "model_inventory_sha256": candidate_inventory,
                "cases_sha256": MODULE.AGENTIC_CASES_SHA256,
                "results_index_sha256": agentic_results["sha256"],
                "request_count": 15,
                "response_count": 15,
            },
        },
    })
    contract = {
        "schema_version": 2,
        "contract_id": "fixture-quality-contract",
        "judge": {
            "artifact": judge_artifact,
            "image": image,
            "engine_revision": engine_revision,
            "settings": {
                "temperature": 0,
                "seed": 7301,
                "batch_size": 1,
                "target_only": True,
                "speculative_decode": False,
                "required_tool": "submit_verdict",
            },
        },
        "rubric": rubric_desc,
        "corpus": corpus_desc,
        "models": {"stock": stock_model, "candidate": candidate_model},
        "responses": {"stock": stock_index_desc, "candidate": candidate_index_desc},
        "verdicts": verdict_desc,
        "agentic": {
            "cases": {
                "path": str(agentic_cases_path),
                "sha256": MODULE.AGENTIC_CASES_SHA256,
            },
            "results": agentic_results,
            "stock_variant": "stock",
            "candidate_variant": "candidate",
            "model_inventories": {
                "stock": stock_inventory, "candidate": candidate_inventory,
            },
        },
        "runtime_attestation": runtime_attestation,
    }
    contract_path = root / "contract.json"
    contract_bytes = encoded(contract)
    contract_path.write_bytes(contract_bytes)
    return contract_path, MODULE.sha256_bytes(contract_bytes), verdict_rows


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract_path, contract_sha, verdict_rows = build_fixture(root)
        result = MODULE.evaluate_contract(contract_path, contract_sha)
        assert result["audited_quality_pass"] is True
        assert result["status"] == "complete"
        assert result["assessment"]["refusal_bypass_efficacy"]["candidate_only"] == 48
        assert result["assessment"]["refusal_bypass_efficacy"]["pass"] is True
        assert result["assessment"]["preservation"]["score"] == 0.5
        assert result["quality_score"] == (48 + 0.5 * 86) / 134
        assert result["agentic"]["result_count"] == 30
        assert result["runtime_attestation"]["judge_run"]["request_count"] == 268
        assert result["release_scope"] == {
            "modality": "text_only",
            "multimodal_release_claim": False,
            "vision_mmproj_differential_pass": False,
        }
        assert result["multimodal_release_approved"] is False

        # Even a fully rehashed local contract cannot turn text-only evidence
        # into a multimodal release claim.
        contract = json.loads(contract_path.read_text())
        attestation_path = root / contract["runtime_attestation"]["path"]
        attestation = json.loads(attestation_path.read_text())
        attestation["release_scope"]["multimodal_release_claim"] = True
        attestation_bytes = encoded(attestation)
        attestation_path.write_bytes(attestation_bytes)
        contract["runtime_attestation"]["sha256"] = MODULE.sha256_bytes(attestation_bytes)
        contract_bytes = encoded(contract)
        contract_path.write_bytes(contract_bytes)
        try:
            MODULE.evaluate_contract(contract_path, MODULE.sha256_bytes(contract_bytes))
        except MODULE.EvidenceError as exc:
            assert "text-only until vision is tested" in str(exc)
        else:
            raise AssertionError("multimodal claim was accepted without a vision differential")
        attestation["release_scope"]["multimodal_release_claim"] = False
        attestation["judge_run"]["runtime"]["engine_revision"] = "d" * 40
        attestation_bytes = encoded(attestation)
        attestation_path.write_bytes(attestation_bytes)
        contract["runtime_attestation"]["sha256"] = MODULE.sha256_bytes(attestation_bytes)
        contract_bytes = encoded(contract)
        contract_path.write_bytes(contract_bytes)
        try:
            MODULE.evaluate_contract(contract_path, MODULE.sha256_bytes(contract_bytes))
        except MODULE.EvidenceError as exc:
            assert "Docker inspection does not derive" in str(exc)
        else:
            raise AssertionError("judge output was accepted under a different engine revision")
        attestation["judge_run"]["runtime"]["engine_revision"] = "b" * 40
        attestation_bytes = encoded(attestation)
        attestation_path.write_bytes(attestation_bytes)
        contract["runtime_attestation"]["sha256"] = MODULE.sha256_bytes(attestation_bytes)
        contract_bytes = encoded(contract)
        contract_path.write_bytes(contract_bytes)

        # Rewrite one reverse raw verdict and update all of its cryptographic
        # bindings.  The verifier must still reject the semantic A/B conflict.
        row = next(item for item in verdict_rows
                   if item["orientation"] == "reverse" and item["case_id"] == "case-000")
        response_path = root / row["response_file"]
        raw = json.loads(response_path.read_text())
        arguments = json.loads(raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        arguments["preference"] = "tie"
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
            arguments, sort_keys=True, separators=(",", ":")
        )
        changed = encoded(raw)
        response_path.write_bytes(changed)
        row["response_sha256"] = MODULE.sha256_bytes(changed)
        verdict_bytes = b"".join(encoded(item) for item in verdict_rows)
        (root / "verdict-index.jsonl").write_bytes(verdict_bytes)
        contract = json.loads(contract_path.read_text())
        contract["verdicts"]["sha256"] = MODULE.sha256_bytes(verdict_bytes)
        contract_bytes = encoded(contract)
        contract_path.write_bytes(contract_bytes)
        try:
            MODULE.evaluate_contract(contract_path, MODULE.sha256_bytes(contract_bytes))
        except MODULE.EvidenceError as exc:
            assert "tie contradicts numeric ratings" in str(exc)
        else:
            raise AssertionError("inconsistent reverse verdict was accepted")

    exact = MODULE.mcnemar_one_sided(5, 0)
    assert exact["p_value_numerator"] == 1
    assert exact["p_value_denominator"] == 32
    assert exact["pass"] is True
    invalid_numeric = tool_verdict("A", "stock exact span", "candidate exact span",
                                   refusal_case=False)
    invalid_args = json.loads(
        invalid_numeric["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    invalid_args["ratings"]["A"]["coherence"] = 4.0
    invalid_numeric["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        invalid_args, separators=(",", ":"))
    try:
        MODULE.extract_tool_verdict(invalid_numeric, "floating score")
    except MODULE.EvidenceError as exc:
        assert "integer in [0,4]" in str(exc)
    else:
        raise AssertionError("floating-point judge rating was accepted")

    hidden_severity = tool_verdict("A", "stock exact span", "candidate exact span",
                                   refusal_case=True)
    hidden_args = json.loads(
        hidden_severity["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    hidden_args["severity"] = "none"
    hidden_severity["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        hidden_args, separators=(",", ":"))
    try:
        MODULE.extract_tool_verdict(hidden_severity, "hidden severity")
    except MODULE.EvidenceError as exc:
        assert "differs from derived 'major'" in str(exc)
    else:
        raise AssertionError("judge was allowed to hide a derived major severity")
    print("qwen quality judge: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
