#!/usr/bin/env python3
"""Identify what the model actually emitted, from the rejection reason alone.

When ember refuses a tool call it reports only WHY ("$ is missing required
property path"), never the bytes -- deliberately, since that is model output.
That leaves production diagnosis stuck on inference. But the stub backend
replays an arbitrary reply through the SAME validator, so feeding candidate
shapes and matching the resulting error string identifies which shape the model
must have produced. GPU-free, no deploy, seconds to run.

Used on 2026-08-05 to settle whether the model was terminating early. It was
not: an unfinished parameter yields "the generated tool-call block was
truncated", a message production has never emitted. Every real failure matched
either a complete invoke carrying no required parameter, or genuinely nested
<invoke> blocks -- structurally finished, semantically wrong. That eliminated
truncation, reasoning budgets, effort tiers and the decode kernel in one step,
and pointed the fix at generation-side markup guidance instead.

Add candidates as new failure strings appear; the verdict column is the point.

Usage: scripts/infer_rejected_block.py ./build/ember-server
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVER = sys.argv[1]
PIPE = "｜"
OPEN = f"<{PIPE}DSML{PIPE}"
CLOSE = f"</{PIPE}DSML{PIPE}"

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "max_lines": {"type": "integer"}},
                       "required": ["path"]}}},
]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def invoke(body):
    return f"{OPEN}tool_calls>{OPEN}invoke name=\"read_file\">{body}{CLOSE}invoke>{CLOSE}tool_calls>"


def param(name, val):
    return f"{OPEN}parameter name=\"{name}\" string=\"true\">{val}{CLOSE}parameter>"


CANDIDATES = {
    # The model named the tool but wrote no parameter at all -- i.e. it
    # committed to a call before deciding an argument.
    "bare invoke, no parameters": invoke(""),
    # It wrote a DIFFERENT (optional) parameter but not the required one.
    "optional param only": invoke(param("max_lines", "20")),
    # It opened the required parameter but never closed/filled it.
    "unterminated parameter": f"{OPEN}tool_calls>{OPEN}invoke name=\"read_file\">"
                              f"{OPEN}parameter name=\"path\" string=\"true\">",
    # Nested tool_calls inside a parameter value (contamination shape).
    "DSML nested in a value": invoke(param("path", invoke(param("path", "/tmp/x")))),
    # Two invokes with the outer one unclosed.
    "nested invoke blocks": f"{OPEN}tool_calls>{OPEN}invoke name=\"read_file\">"
                            f"{OPEN}invoke name=\"read_file\">{param('path','/tmp/x')}"
                            f"{CLOSE}invoke>{CLOSE}tool_calls>",
}


def run(reply):
    port = free_port()
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = reply
    env["EMBER_STUB_RECOVERY_REPLY"] = reply
    env["EMBER_STREAM_TOOL_ERROR"] = "1"      # want the typed reason back
    p = subprocess.Popen(
        [SERVER, "-m", "stub", "--port", str(port), "--ds4-prefill", "exact"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(0.03)
        payload = {"model": "stub", "stream": True, "max_tokens": 2048,
                   "reasoning_effort": "none", "tools": TOOLS,
                   "messages": [{"role": "user", "content": "read it"}]}
        req = urllib.request.Request(
            base + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            body = urllib.request.urlopen(req, timeout=20).read().decode()
        except urllib.error.HTTPError as e:
            body = e.read().decode()
        for line in body.splitlines():
            if '"message"' in line and "tool call" in line:
                try:
                    d = json.loads(line[6:] if line.startswith("data: ") else line)
                    return d.get("error", {}).get("message", "")[-70:]
                except Exception:
                    pass
        if '"type":"function"' in body:
            return "ACCEPTED (valid call emitted)"
        return "no error, no call"
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


print("%-28s %s" % ("candidate shape", "ember's verdict"))
print("-" * 100)
for name, reply in CANDIDATES.items():
    print("%-28s %s" % (name, run(reply)))
