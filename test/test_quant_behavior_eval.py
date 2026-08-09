#!/usr/bin/env python3
"""Unit checks for quant behavioral result classification."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quant_behavior_eval", ROOT / "scripts" / "quant_behavior_eval.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> int:
    case = {
        "id": "write",
        "category": "tool_use",
        "messages": [{"role": "user", "content": "write it"}],
        "expect": {
            "tool_name": "write_file",
            "tool_arguments": {"path": "/tmp/x", "content": "hello\n"},
            "finish_reason": "tool_calls",
        },
    }
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path":"/tmp/x","content":"hello\\n"}',
                            }
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "timings": {"prefill_tokens_per_sec": 100, "decode_tokens_per_sec": 20},
        },
    }
    result = MODULE.classify(case, response, 0.5)
    assert result["success"] is True
    assert result["tool_call_valid"] is True
    assert result["decode_tokens_per_second"] == 20

    response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{}"
    result = MODULE.classify(case, response, 0.5)
    assert result["success"] is False
    assert any("tool arguments" in error for error in result["errors"])

    leak_case = {"id": "leak", "messages": [{"role": "user", "content": "x"}]}
    leak_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "<｜DSML｜function_calls> nested output"},
            }
        ]
    }
    result = MODULE.classify(leak_case, leak_response, 1.0)
    assert result["dsml_leak_detected"] is True
    assert result["success"] is False
    assert MODULE.repeated_ngram_fraction("short text") == 0.0

    payload_case = {
        "id": "route",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 17,
        "request": {"seed": 7301},
    }
    payload = MODULE.build_payload(
        payload_case,
        "deepseek/deepseek-v4-flash-0731",
        {"provider": {"order": ["deepseek"], "allow_fallbacks": False}},
        0.0,
    )
    assert payload["provider"]["order"] == ["deepseek"]
    assert payload["seed"] == 7301
    assert payload["max_tokens"] == 17

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "request.json"
        path.write_text(json.dumps({"provider": "DeepSeek"}), encoding="utf-8")
        assert MODULE.load_object(path) == {"provider": "DeepSeek"}
    print("quant behavior eval: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
