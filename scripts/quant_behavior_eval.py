#!/usr/bin/env python3
"""Run deterministic behavioral replay cases against an OpenAI chat endpoint.

The output JSONL is consumed by quant_quality_report.py.  Run the same case set
once with ``--variant reference`` and once with ``--variant quant --append``.
Raw responses are retained beside the JSONL by default so every classification
can be audited rather than inferred from a pass percentage alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: {exc}") from exc
            if not isinstance(case, dict) or not case.get("id") or not case.get("messages"):
                raise SystemExit(f"{path}:{line_no}: case requires id and messages")
            cases.append(case)
    return cases


def load_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return value


def build_payload(
    case: dict[str, Any], model: str, request_overrides: dict[str, Any], temperature: float
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": case["messages"],
        "stream": False,
        "temperature": temperature,
        "max_tokens": int(case.get("max_tokens", 256)),
    }
    payload.update(request_overrides)
    if case.get("tools") is not None:
        payload["tools"] = case["tools"]
    payload.update(case.get("request") or {})
    return payload


def post_json(
    endpoint: str, payload: dict[str, Any], api_key: str | None, timeout: float, retries: int
) -> tuple[dict[str, Any], int]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    delay = 1.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8")), response.status
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", "replace")
            last = RuntimeError(f"HTTP {exc.code}: {error_body}")
            if exc.code < 500 and exc.code != 429:
                break
        except Exception as exc:  # noqa: BLE001 - CLI retry boundary.
            last = exc
        if attempt < retries:
            time.sleep(delay)
            delay = min(10.0, delay * 1.7)
    assert last is not None
    raise last


def repeated_ngram_fraction(text: str, n: int = 8) -> float:
    words = re.findall(r"\S+", text.casefold())
    if len(words) < max(32, n * 2):
        return 0.0
    windows = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not windows:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for window in windows:
        counts[window] = counts.get(window, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(windows)


# Keep the corpus verdict aligned with the marker families Ember actually
# parses (tool_parser.c:12-28). A singular illustrative tag such as
# ``<tool_call>`` is not an executable Ember marker and must not be reported as
# a raw DSML leak merely because a response discusses protocol design.
RAW_TOOL_MARKUP_RE = re.compile(
    r"</?(?:(?:｜DSML｜|DSML｜|\?DSML\?)|"
    r"(?:tool_calls|invoke|parameter|ds_engine_tool_use)(?:\s|>))",
    re.IGNORECASE,
)


def contains_raw_tool_markup(text: str) -> bool:
    return bool(RAW_TOOL_MARKUP_RE.search(text))


def tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    return calls if isinstance(calls, list) else []


def valid_tool_calls(calls: list[dict[str, Any]]) -> bool | None:
    if not calls:
        return None
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return False
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return False
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
    return True


def classify(case: dict[str, Any], response: dict[str, Any], latency: float) -> dict[str, Any]:
    errors: list[str] = []
    choices = response.get("choices")
    response_valid = isinstance(choices, list) and bool(choices) and isinstance(choices[0], dict)
    choice = choices[0] if response_valid else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    reasoning = message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else ""
    calls = tool_calls(message)
    tool_valid = valid_tool_calls(calls)
    finish_reason = choice.get("finish_reason")
    expected = case.get("expect") if isinstance(case.get("expect"), dict) else {}

    if not response_valid:
        errors.append("invalid OpenAI response envelope")
    for needle in expected.get("contains", []):
        if str(needle).casefold() not in content.casefold():
            errors.append(f"content missing {needle!r}")
    for pattern in expected.get("regex", []):
        if not re.search(str(pattern), content, flags=re.IGNORECASE | re.MULTILINE):
            errors.append(f"content does not match /{pattern}/")
    if "content_json_equals" in expected:
        try:
            content_json = json.loads(content)
        except json.JSONDecodeError:
            content_json = None
            errors.append("visible content is not valid JSON")
        if content_json is not None and content_json != expected["content_json_equals"]:
            errors.append("visible JSON does not equal the expected value")

    expected_tool = expected.get("tool_name")
    names = [
        call.get("function", {}).get("name")
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("function"), dict)
    ]
    if expected_tool and expected_tool not in names:
        errors.append(f"expected tool {expected_tool!r}, got {names!r}")
    expected_arguments = expected.get("tool_arguments")
    if expected_tool and expected_arguments is not None and expected_tool in names:
        call = calls[names.index(expected_tool)]
        try:
            actual_arguments = json.loads(call["function"]["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            actual_arguments = None
        if actual_arguments != expected_arguments:
            errors.append(
                f"tool arguments {actual_arguments!r} do not equal {expected_arguments!r}"
            )
    if expected.get("no_tool") and calls:
        errors.append(f"unexpected tool call(s): {names!r}")
    if calls and tool_valid is not True:
        errors.append("malformed tool call")
    expected_finish = expected.get("finish_reason")
    if expected_finish and finish_reason != expected_finish:
        errors.append(f"finish_reason={finish_reason!r}, expected {expected_finish!r}")

    identifiers = [str(value) for value in expected.get("identifiers", [])]
    identifier_integrity = all(value in content for value in identifiers) if identifiers else None
    if identifier_integrity is False:
        errors.append("one or more required identifiers changed or disappeared")

    # Visible content and hidden reasoning are separate channels. Models often
    # draft the final sentence once in reasoning and then emit it once in
    # content; concatenating the channels misclassifies that normal handoff as
    # an 8-gram loop. Gate on degeneration within either channel instead.
    repeat_fraction = max(
        repeated_ngram_fraction(content),
        repeated_ngram_fraction(reasoning),
    )
    repeat_limit = float(expected.get("max_repeated_ngram_fraction", 0.20))
    repetition_detected = repeat_fraction > repeat_limit
    if repetition_detected:
        errors.append(f"repeated 8-gram fraction {repeat_fraction:.3f} exceeds {repeat_limit:.3f}")

    dsml_leak = contains_raw_tool_markup(content)
    thinking_leak = "<think>" in content or "</think>" in content
    if dsml_leak:
        errors.append("raw DSML marker leaked into visible content")
    if thinking_leak:
        errors.append("thinking marker leaked into visible content")

    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "case_id": case["id"],
        "category": case.get("category", "uncategorized"),
        "success": not errors,
        "response_valid": response_valid,
        "tool_call_valid": tool_valid,
        "identifier_integrity": identifier_integrity,
        "repetition_detected": repetition_detected,
        "repeated_ngram_fraction": repeat_fraction,
        "dsml_leak_detected": dsml_leak,
        "thinking_leak_detected": thinking_leak,
        "finish_reason": finish_reason,
        "latency_seconds": latency,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion_tokens,
        "tokens_per_second": completion_tokens / latency if completion_tokens and latency > 0 else None,
        "prefill_tokens_per_second": (
            (usage.get("timings") or {}).get("prefill_tokens_per_sec")
            if isinstance(usage.get("timings"), dict)
            else None
        ),
        "decode_tokens_per_second": (
            (usage.get("timings") or {}).get("decode_tokens_per_sec")
            if isinstance(usage.get("timings"), dict)
            else None
        ),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="full /v1/chat/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=("reference", "quant"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--request-overrides",
        type=Path,
        help="JSON object merged into every request (for example a pinned provider route)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    raw_dir = args.raw_dir or args.out.with_suffix(args.out.suffix + ".responses")
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    request_overrides = load_object(args.request_overrides)

    failures = 0
    mode = "a" if args.append else "w"
    with args.out.open(mode, encoding="utf-8") as output:
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {args.variant} {case['id']}", file=sys.stderr, flush=True)
            payload = build_payload(case, args.model, request_overrides, args.temperature)
            started = time.monotonic()
            try:
                response, status = post_json(
                    args.endpoint, payload, api_key, args.timeout, max(0, args.retries)
                )
                latency = time.monotonic() - started
                result = classify(case, response, latency)
                result["http_status"] = status
            except Exception as exc:  # noqa: BLE001 - one case must not abort the suite.
                latency = time.monotonic() - started
                response = {"error": str(exc)}
                result = {
                    "case_id": case["id"],
                    "category": case.get("category", "uncategorized"),
                    "success": False,
                    "response_valid": False,
                    "tool_call_valid": None,
                    "identifier_integrity": None,
                    "repetition_detected": None,
                    "dsml_leak_detected": None,
                    "thinking_leak_detected": None,
                    "latency_seconds": latency,
                    "errors": [str(exc)],
                    "http_status": None,
                }
            raw_path = raw_dir / f"{args.variant}_{case['id']}.json"
            raw_bytes = (json.dumps(response, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            raw_path.write_bytes(raw_bytes)
            result.update(
                {
                    "variant": args.variant,
                    "model": args.model,
                    "endpoint": args.endpoint,
                    "response_file": str(raw_path.resolve()),
                    "response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                }
            )
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            if not result["success"]:
                failures += 1
                print(f"  FAIL: {'; '.join(result['errors'])}", file=sys.stderr)
    print(f"behavior replay: {len(cases) - failures}/{len(cases)} passed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
