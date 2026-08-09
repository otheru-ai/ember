#!/usr/bin/env python3
"""Exercise the wire contracts used by popular coding-agent clients."""

import json
import os
import subprocess
import sys

from test_tool_safety_server import free_port, request, wait_ready


def require_events(stream: str, *events: str) -> None:
    position = 0
    for event in events:
        found = stream.find(event, position)
        assert found >= 0, f"missing or out-of-order {event!r}: {stream}"
        position = found + len(event)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/ember-server", file=sys.stderr)
        return 2

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "Hello from Ember."
    proc = subprocess.Popen(
        [sys.argv[1], "-m", "stub", "--port", str(port),
         "--ds4-prefill", "exact"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base, proc)

        # Claude Code: Anthropic Messages, including cache-control annotations
        # that a local gateway may safely ignore and native SSE event order.
        claude = {
            "model": "deepseek-v4-flash",
            "max_tokens": 1024,
            "stream": True,
            "thinking": {"type": "disabled"},
            "system": [{
                "type": "text",
                "text": "You are a coding agent.",
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Say hello.",
                    "cache_control": {"type": "ephemeral"},
                }],
            }],
            "tools": [{
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }],
            "tool_choice": {"type": "auto"},
        }
        code, stream = request(base + "/v1/messages", claude)
        assert code == 200, stream
        assert isinstance(stream, str), stream
        require_events(
            stream,
            "event: message_start",
            "event: content_block_start",
            '"type":"text_delta"',
            "event: content_block_stop",
            "event: message_delta",
            "event: message_stop",
        )

        # Codex: Responses with the fields and function-tool shape emitted by
        # the CLI. Hosted web_search must be disabled client-side because it is
        # an OpenAI-executed tool, not a local function call.
        codex = {
            "model": "deepseek-v4-flash",
            "instructions": "You are a coding agent.",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello."}],
            }],
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
                "strict": False,
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"effort": "none", "summary": "auto"},
            "store": False,
            "stream": True,
        }
        code, stream = request(base + "/v1/responses", codex)
        assert code == 200, stream
        assert isinstance(stream, str), stream
        require_events(
            stream,
            '"type":"response.created"',
            '"type":"response.output_item.added"',
            '"type":"response.output_text.delta"',
            '"type":"response.output_text.done"',
            '"type":"response.output_item.done"',
            '"type":"response.completed"',
        )

        # OpenCode, pi, and OMP all support OpenAI Chat Completions. Exercise
        # developer messages, content parts, function tools, and usage-bearing
        # streaming: the intersection used by their local-provider adapters.
        openai_compatible = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "developer", "content": "You are a coding agent."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Say hello."}
                ]},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning_effort": "none",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        code, stream = request(base + "/v1/chat/completions", openai_compatible)
        assert code == 200, stream
        assert isinstance(stream, str), stream
        require_events(stream, '"role":"assistant"', '"content":"H"',
                       '"usage":{', "data: [DONE]")

        code, models = request(base + "/v1/models")
        assert code == 200, models
        assert models["data"][0]["id"] == "deepseek-v4-flash", models

        print("coding-client compatibility: PASS")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
