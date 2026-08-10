#!/usr/bin/env python3
"""Tool-calling conformance suite (GPU-free half is test/test_tool_parser.c).

Measures a MALFORMED-CALL RATE, not "did a tool fire". Every emitted call is
validated against its declared JSON schema: arguments must parse, required
properties must be present with the right types, enums must be respected, and
nested objects recurse. That precision is the point -- the production failures
this was written for look like

    "unsafe or incomplete tool call ... $ is missing required property command"

which a naive `grep tool_calls` scores as a pass.

Both transports are exercised. Weight the streaming half: ember deliberately
does not hide a replacement assistant attempt inside an open response
(CLAUDE.md), so a malformed call is silently recovered on the buffered path but
surfaces as a hard turn failure on the streaming path -- which is the one a
streaming agent actually uses.

Usage: qa_toolcalls.py <port> <tag> <reps>
Prints one FAIL line per defect and a final TOOLSUITE summary; exits nonzero if
any defect was seen.
"""
import json
import sys
import urllib.request

PORT, TAG, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
URL = "http://127.0.0.1:%s/v1/chat/completions" % PORT

TOOLS = [
    {"type": "function", "function": {
        "name": "terminal",
        "description": "Run a shell command and return its output.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "max_lines": {"type": "integer"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Search the repository for a pattern.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "limit": {"type": "integer"},
                                      "extensions": {"type": "array",
                                                     "items": {"type": "string"}}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_task",
        "description": "Create a task with metadata.",
        "parameters": {"type": "object",
                       "properties": {
                           "title": {"type": "string"},
                           "metadata": {"type": "object",
                                        "properties": {
                                            "priority": {"type": "string",
                                                         "enum": ["low", "high"]},
                                            "tags": {"type": "array",
                                                     "items": {"type": "string"}}},
                                        "required": ["priority"]}},
                       "required": ["title", "metadata"]}}},
]
BY_NAME = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

# (prompt, expected tool or None). The None cases are the over-trigger guard:
# a greeting offered a terminal tool must not call it.
CASES = [
    ("List the files in /etc using the terminal tool.", "terminal"),
    ("Use the terminal to show current disk usage.", "terminal"),
    ("Run `uname -a` for me.", "terminal"),
    ("Read the file /etc/hostname.", "read_file"),
    ("Read the first 20 lines of /var/log/syslog.", "read_file"),
    ("Search the code for the pattern 'malloc' in .c files.", "search_code"),
    ("Find where 'ember_backend_generate' is defined.", "search_code"),
    ("Create a high priority task titled 'ship release' tagged infra.", "create_task"),
    ("Hello, how are you today?", None),
    ("Thanks, that's all I needed.", None),
]

TYPES = {"string": str, "integer": int, "number": (int, float),
         "array": list, "object": dict, "boolean": bool}


def check_args(schema, args, path=""):
    """Recursive JSON-Schema conformance. Returns a list of violations."""
    bad = []
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            bad.append("missing required property %s%s" % (path, req))
    for k, v in args.items():
        if k not in props:
            bad.append("unknown property %s%s" % (path, k))
            continue
        want = props[k].get("type")
        if want in TYPES and not isinstance(v, TYPES[want]):
            bad.append("wrong type for %s%s: want %s got %s"
                       % (path, k, want, type(v).__name__))
        if want == "object" and isinstance(v, dict):
            bad.extend(check_args(props[k], v, path + k + "."))
        if want == "array" and isinstance(v, list):
            it = props[k].get("items", {}).get("type")
            if it in TYPES and not all(isinstance(x, TYPES[it]) for x in v):
                bad.append("array %s%s has wrong element type" % (path, k))
        if "enum" in props[k] and v not in props[k]["enum"]:
            bad.append("value %r not in enum for %s%s" % (v, path, k))
    return bad


def one(prompt, expect, stream):
    body = {"model": "deepseek-v4-flash", "tools": TOOLS,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0.6, "stream": stream}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        raw = urllib.request.urlopen(req, timeout=600).read().decode()
    except Exception as e:
        return ["transport error: %s" % e]

    if stream:
        # Reassemble tool calls from SSE deltas; arguments arrive fragmented.
        calls = {}
        for line in raw.splitlines():
            if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            for ch in d.get("choices", []):
                for tc in ch.get("delta", {}).get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0),
                                            {"name": "", "args": ""})
                    fn = tc.get("function", {})
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
        emitted = [(c["name"], c["args"]) for c in calls.values()]
    else:
        msg = json.loads(raw)["choices"][0]["message"]
        emitted = [(tc["function"]["name"], tc["function"]["arguments"])
                   for tc in (msg.get("tool_calls") or [])]

    if expect is None:
        return (["over-trigger: fired %s on a non-tool prompt"
                 % [{"name": name, "arguments": args}
                    for name, args in emitted]] if emitted else [])
    if not emitted:
        return ["no tool call emitted (expected %s)" % expect]

    bad = []
    for name, argstr in emitted:
        if name not in BY_NAME:
            bad.append("unknown tool %r" % name)
            continue
        try:
            args = json.loads(argstr) if argstr.strip() else {}
        except Exception as e:
            bad.append("arguments not valid JSON (%s): %.80r" % (e, argstr))
            continue
        bad.extend(check_args(BY_NAME[name], args))
    return bad


total = defects = 0
by_kind = {}
for rep in range(REPS):
    for idx, (prompt, expect) in enumerate(CASES):
        for stream in (False, True):
            bad = one(prompt, expect, stream)
            total += 1
            if bad:
                defects += 1
                for b in bad:
                    kind = b.split(":")[0].split("(")[0].strip()
                    by_kind[kind] = by_kind.get(kind, 0) + 1
                    print("FAIL [%s] rep%d case%d stream=%d: %s"
                          % (TAG, rep, idx, stream, b), flush=True)

print("TOOLSUITE %s total=%d defects=%d rate=%.4f breakdown=%s"
      % (TAG, total, defects, (defects / total) if total else 0.0,
         json.dumps(by_kind)), flush=True)
sys.exit(1 if defects else 0)
