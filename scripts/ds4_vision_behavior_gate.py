#!/usr/bin/env python3
"""Run one digest-bound DeepSeek-V4 Vision two-arm behavioural gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


POLICY_SCHEMA = "ds4-vision-behavioural-gate-policy-v4"
POLICY_SHA256 = "b82d86ff9989f6c57656ac868b4eeb601d8009ce784c666124bbca0535c4bb03"
EVIDENCE_SCHEMA = "ember.ds4.vision-behavioural-evidence.v1"
RUNTIME_SCHEMA = "ember.ds4.vision-behavioural-runtime.v1"
CORPUS_SHA256 = {
    "synthetic": "f1b8303e279899257e61c5311084bb6fd836aeadc88dcebe9da0b7246dff5b98",
    "natural": "b69b1979e4cee4fa294e0b759cba62dd715f6ae86376ccf2626eb72e872f325c",
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
ITEM_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class GateError(ValueError):
    pass


class RepetitionRefusal(Exception):
    """Ember refused a degenerate generation with HTTP 422.

    Policy v4 scores this asymmetrically, and the asymmetry is the point. In the
    no-image arm, degenerating IS the measurement -- that arm exists to ask
    whether the model can answer without seeing the picture, and looping instead
    of answering is a clear no. Treating it as a transport error once discarded
    an entire 100-item calibration over the single item that demonstrated the
    point best. With the image present it stays an error, because a model that
    degenerates while looking at the picture is a real failure and must not be
    quietly filed as one wrong answer.
    """

    def __init__(self, raw: bytes):
        super().__init__("generation refused as degenerate")
        self.raw = raw


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stable_bytes(path: Path, label: str, maximum: int) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise GateError(f"{label} must be an absolute non-symlink file")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise GateError(f"{label} must be a non-empty regular file")
    if before.st_size > maximum:
        raise GateError(f"{label} exceeds its byte ceiling")
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise GateError(f"{label} changed while read")
    return raw


def read_object(path: Path, label: str, maximum: int) -> tuple[dict[str, Any], bytes]:
    raw = stable_bytes(path, label, maximum)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value, raw


def write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def descriptor(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": len(raw),
            "sha256": sha256_bytes(raw)}


def validate_policy(path: Path) -> tuple[dict[str, Any], bytes]:
    policy, raw = read_object(path, "vision gate policy", 128 * 1024)
    if sha256_bytes(raw) != POLICY_SHA256:
        raise GateError("vision gate policy is not the preregistered v4 policy")
    if (policy.get("schema") != POLICY_SCHEMA or
            policy.get("decided_by") != "user" or
            policy.get("package") != "balanced"):
        raise GateError("vision gate policy identity differs")
    generation = policy.get("generation")
    scoring = policy.get("scoring")
    synthetic = policy.get("synthetic")
    natural = policy.get("natural")
    if (not isinstance(generation, dict) or
            generation.get("decoding") != "greedy" or
            generation.get("temperature") != 0.0 or
            generation.get("max_tokens") != 512 or
            not isinstance(scoring, dict) or
            not isinstance(synthetic, dict) or
            not isinstance(natural, dict)):
        raise GateError("vision gate policy execution contract differs")
    if (synthetic.get("n_per_class") != 25 or
            synthetic.get("classes") != ["count", "colour", "spatial", "ocr"] or
            synthetic.get("arm_a_min_accuracy_per_class") != 0.90 or
            (synthetic.get("arm_a_significance") or {}).get("alpha") != 0.01 or
            (synthetic.get("arm_b_requirement") or {}).get("alpha") != 0.05 or
            natural.get("n_items") != 100 or
            natural.get("min_retained_after_cuts") != 80 or
            natural.get("arm_a_min_accuracy") != 0.70 or
            natural.get("max_tokens") != 2048):
        raise GateError("vision gate policy thresholds differ")
    return policy, raw


def validate_runtime(path: Path, expected_sha256: str,
                     endpoint: str, model: str) -> tuple[dict[str, Any], bytes]:
    parsed = urllib.parse.urlparse(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"} or
            parsed.path != "/v1/chat/completions" or parsed.params or
            parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise GateError("behavioural endpoint must be the local chat-completions path")
    if HEX64.fullmatch(expected_sha256) is None:
        raise GateError("runtime identity expected SHA-256 is malformed")
    value, raw = read_object(path, "runtime identity", 1024 * 1024)
    if sha256_bytes(raw) != expected_sha256:
        raise GateError("runtime identity SHA-256 differs")
    if set(value) != {"schema", "ember_commit", "engine_binary_sha256",
                      "model_identity_sha256", "vision_mmproj_sha256",
                      "gpu_arch", "rocm_version", "endpoint", "model",
                      "batch_sessions", "exact_prefill", "speculation"}:
        raise GateError("runtime identity fields differ")
    if (value.get("schema") != RUNTIME_SCHEMA or
            HEX40.fullmatch(str(value.get("ember_commit"))) is None or
            HEX64.fullmatch(str(value.get("engine_binary_sha256"))) is None or
            HEX64.fullmatch(str(value.get("model_identity_sha256"))) is None or
            HEX64.fullmatch(str(value.get("vision_mmproj_sha256"))) is None or
            value.get("gpu_arch") != "gfx1151" or
            not isinstance(value.get("rocm_version"), str) or
            not value["rocm_version"] or
            value.get("endpoint") != endpoint or value.get("model") != model or
            value.get("batch_sessions") != 1 or
            value.get("exact_prefill") is not False or
            value.get("speculation") is not False):
        raise GateError("runtime identity contract differs")
    return value, raw


def corpus_root(path: Path, expected: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise GateError("corpus must be an absolute non-symlink directory")
    if str(path) != expected:
        raise GateError("corpus path differs from the preregistered policy")
    return path


def load_corpus(kind: str, root: Path, policy: dict[str, Any]) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    contract = policy[kind]
    root = corpus_root(root, contract["corpus"])
    manifest, manifest_raw = read_object(
        (root / "manifest.json").absolute(), f"{kind} corpus manifest", 4 * 1024 * 1024)
    if sha256_bytes(manifest_raw) != CORPUS_SHA256[kind]:
        raise GateError(f"{kind} corpus manifest identity differs")
    items = manifest.get("items")
    expected_n = 100
    if (manifest.get("n_items") != expected_n or
            not isinstance(items, list) or len(items) != expected_n):
        raise GateError(f"{kind} corpus size differs")

    classes = contract["classes"] if kind == "synthetic" else ["textvqa"]
    counts = {name: 0 for name in classes}
    seen: set[str] = set()
    loaded = []
    for item in items:
        if not isinstance(item, dict):
            raise GateError(f"{kind} corpus item is not an object")
        item_id = item.get("id")
        item_class = item.get("class")
        rel_image = item.get("image")
        question = item.get("question")
        answer = item.get("answer")
        chance = item.get("chance")
        digest = item.get("sha256")
        if (not isinstance(item_id, str) or ITEM_ID.fullmatch(item_id) is None or
                item_id in seen or item_class not in counts or
                not isinstance(rel_image, str) or
                rel_image != f"images/{item_id}.png" or
                not isinstance(question, str) or not question.strip() or
                not isinstance(answer, str) or not answer.strip() or
                not isinstance(chance, (int, float)) or isinstance(chance, bool) or
                not math.isfinite(float(chance)) or float(chance) < 0.0 or
                float(chance) >= 1.0 or HEX64.fullmatch(str(digest)) is None):
            raise GateError(f"{kind} corpus item is malformed")
        image_path = (root / rel_image).absolute()
        image = stable_bytes(image_path, f"image {item_id}", 64 * 1024 * 1024)
        if sha256_bytes(image) != digest or not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise GateError(f"image {item_id} identity differs")
        seen.add(item_id)
        counts[item_class] += 1
        loaded.append({"id": item_id, "class": item_class,
                       "question": question, "answer": answer,
                       "chance": float(chance), "image_path": image_path,
                       "image": image, "image_sha256": digest})
    if kind == "synthetic":
        if any(count != contract["n_per_class"] for count in counts.values()):
            raise GateError("synthetic class coverage differs")
        for name in classes:
            chances = {row["chance"] for row in loaded if row["class"] == name}
            if len(chances) != 1 or next(iter(chances)) <= 0.0:
                raise GateError("synthetic class chance is not constant and positive")
    elif counts != {"textvqa": contract["n_items"]}:
        raise GateError("natural class coverage differs")
    return manifest, manifest_raw, loaded


def normalize(text: str) -> str:
    stripped = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return " ".join(stripped.split())


def answer_matches(response: str, answer: str) -> bool:
    actual = normalize(response)
    wanted = normalize(answer)
    if not actual or not wanted:
        return False
    return actual == wanted or f" {wanted} " in f" {actual} "


def binomial_upper_tail(successes: int, trials: int, chance: float) -> float:
    if trials <= 0 or successes < 0 or successes > trials or not 0.0 <= chance <= 1.0:
        raise GateError("invalid exact-binomial inputs")
    return sum(math.comb(trials, k) * chance**k * (1.0 - chance)**(trials - k)
               for k in range(successes, trials + 1))


def generation_budget(policy: dict[str, Any], kind: str) -> int:
    """The completion budget this kind runs under, taken from the policy.

    It used to be the literal 32 here while the policy was merely asserted to
    say 32, which is the failure the policy exists to prevent: a threshold the
    runner does not read is not a threshold. This model emits reasoning_content
    before content and reasoning spends the same budget, so 32 returned
    finish_reason "length" with empty content on every item -- red for a harness
    reason with no bearing on the model. The natural set overrides the default
    again because its answers are longer.
    """
    override = policy[kind].get("max_tokens")
    return int(override if override is not None else policy["generation"]["max_tokens"])


def build_payload(row: dict[str, Any], model: str, arm: str,
                  max_tokens: int) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if arm == "A":
        content.append({"type": "image_url", "image_url": {"url":
            "data:image/png;base64," + base64.b64encode(row["image"]).decode("ascii")}})
    elif arm != "B":
        raise GateError("unknown behavioural arm")
    content.append({"type": "text", "text": row["question"]})
    return {"model": model, "stream": False, "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}]}


def post_json(endpoint: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], bytes]:
    wire = canonical(payload)
    request = urllib.request.Request(endpoint, data=wire,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise GateError("behavioural response exceeds its byte ceiling")
            if response.status != 200:
                raise GateError(f"behavioural request returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        if exc.code == 422 and b"repetition_detected" in raw:
            raise RepetitionRefusal(raw) from exc
        raise GateError(f"behavioural request returned HTTP {exc.code}: " +
                        raw.decode("utf-8", "replace")) from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"behavioural response is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError("behavioural response must be a JSON object")
    return value, raw


def response_content(value: dict[str, Any]) -> str:
    choices = value.get("choices")
    if (not isinstance(choices, list) or len(choices) != 1 or
            not isinstance(choices[0], dict) or
            not isinstance(choices[0].get("message"), dict) or
            not isinstance(choices[0]["message"].get("content"), str)):
        raise GateError("behavioural response envelope differs")
    return choices[0]["message"]["content"]


def score_synthetic(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    contract = policy["synthetic"]
    results = []
    first_red = None
    for name in contract["classes"]:
        group = [row for row in rows if row["class"] == name]
        cut = [row["id"] for row in group if row["arm_b_correct"]]
        retained = [row for row in group if not row["arm_b_correct"]]
        chance = group[0]["chance"]
        a_correct = sum(row["arm_a_correct"] for row in retained)
        b_correct = sum(row["arm_b_correct"] for row in group)
        a_accuracy = a_correct / len(retained) if retained else 0.0
        a_p = binomial_upper_tail(a_correct, len(retained), chance)
        b_p = binomial_upper_tail(b_correct, len(group), chance)
        checks = {
            "arm_a_accuracy": a_accuracy >= contract["arm_a_min_accuracy_per_class"],
            "arm_a_significance": a_p <= contract["arm_a_significance"]["alpha"],
            "arm_b_at_chance": b_p > contract["arm_b_requirement"]["alpha"],
        }
        passed = all(checks.values())
        if not passed and first_red is None:
            first_red = name
        results.append({"class": name, "total": len(group),
                        "retained": len(retained), "cut_ids": cut,
                        "chance": chance, "arm_a_correct": a_correct,
                        "arm_a_accuracy": a_accuracy,
                        "arm_a_upper_tail_p": a_p,
                        "arm_b_correct": b_correct,
                        "arm_b_accuracy": b_correct / len(group),
                        "arm_b_upper_tail_p": b_p,
                        "checks": checks, "passed": passed})
    return {"passed": first_red is None, "first_red_class": first_red,
            "classes": results}


def score_natural(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    contract = policy["natural"]
    cut = [row["id"] for row in rows if row["arm_b_correct"]]
    retained = [row for row in rows if not row["arm_b_correct"]]
    a_correct = sum(row["arm_a_correct"] for row in retained)
    accuracy = a_correct / len(retained) if retained else 0.0
    checks = {"retained_count": len(retained) >= contract["min_retained_after_cuts"],
              "arm_a_accuracy": accuracy >= contract["arm_a_min_accuracy"]}
    first_red = next((name for name, passed in checks.items() if not passed), None)
    return {"passed": first_red is None, "first_red": first_red,
            "total": len(rows), "retained": len(retained), "cut_ids": cut,
            "arm_a_correct": a_correct, "arm_a_accuracy": accuracy,
            "checks": checks}


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy, policy_raw = validate_policy(args.policy.absolute())
    identity, identity_raw = validate_runtime(
        args.runtime_identity.absolute(), args.runtime_identity_sha256,
        args.endpoint, args.model)
    _manifest, manifest_raw, rows = load_corpus(
        args.kind, args.corpus.absolute(), policy)

    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    policy_path = output / "policy.json"
    identity_path = output / "runtime-identity.json"
    corpus_path = output / "corpus-manifest.json"
    write_new(policy_path, policy_raw)
    write_new(identity_path, identity_raw)
    write_new(corpus_path, manifest_raw)

    budget = generation_budget(policy, args.kind)
    scored = []
    # Run the withheld control first so no image output is observed before the
    # control arm has been fixed and retained for every item.
    for arm in ("B", "A"):
        for index, row in enumerate(rows, 1):
            print(f"[{arm} {index}/{len(rows)}] {row['id']}", file=sys.stderr, flush=True)
            payload = build_payload(row, args.model, arm, budget)
            request_raw = canonical(payload)
            request_path = output / f"arm-{arm.lower()}" / f"{row['id']}.request.json"
            response_path = output / f"arm-{arm.lower()}" / f"{row['id']}.response.json"
            write_new(request_path, request_raw)
            try:
                value, response_raw = post_json(args.endpoint, payload, args.timeout)
            except RepetitionRefusal as refusal:
                if arm == "A":
                    raise GateError(
                        f"item {row['id']} degenerated WITH the image present; "
                        "policy v4 keeps that an error, not a wrong answer") from refusal
                write_new(response_path, refusal.raw)
                row["arm_b_content"] = ""
                row["arm_b_correct"] = False
                row["arm_b_refused"] = True
                row["arm_b_response"] = descriptor(response_path, refusal.raw)
                continue
            write_new(response_path, response_raw)
            content = response_content(value)
            row[f"arm_{arm.lower()}_content"] = content
            row[f"arm_{arm.lower()}_correct"] = answer_matches(content, row["answer"])
            row[f"arm_{arm.lower()}_refused"] = False
            row[f"arm_{arm.lower()}_response"] = descriptor(response_path, response_raw)

    for row in rows:
        scored.append({"id": row["id"], "class": row["class"],
                       "answer": normalize(row["answer"]), "chance": row["chance"],
                       "image_sha256": row["image_sha256"],
                       "arm_a_normalized": normalize(row["arm_a_content"]),
                       "arm_a_correct": row["arm_a_correct"],
                       "arm_a_response": row["arm_a_response"],
                       "arm_b_normalized": normalize(row["arm_b_content"]),
                       "arm_b_correct": row["arm_b_correct"],
                       "arm_b_refused": row.get("arm_b_refused", False),
                       "arm_b_response": row["arm_b_response"],
                       "cut": row["arm_b_correct"]})

    verdict = (score_synthetic(rows, policy) if args.kind == "synthetic"
               else score_natural(rows, policy))
    result = {"schema": EVIDENCE_SCHEMA, "kind": args.kind,
              "passed": verdict["passed"],
              "policy": descriptor(policy_path, policy_raw),
              "runtime_identity": descriptor(identity_path, identity_raw),
              "corpus": descriptor(corpus_path, manifest_raw),
              "runtime": identity, "generation": policy["generation"],
              "max_tokens": budget,
              "verdict": verdict, "items": scored}
    write_new(output / "result.json", canonical(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("synthetic", "natural"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--runtime-identity-sha256", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    try:
        result = run(args)
    except (GateError, OSError) as exc:
        print(f"vision behavioural gate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": result["passed"],
                      "kind": result["kind"],
                      "output": str(args.output.absolute())}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
