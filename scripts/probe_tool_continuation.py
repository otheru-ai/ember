#!/usr/bin/env python3
"""Drive ember through a REAL multi-tool agent loop.

Flattened conversation replay was clean (0/8), because it re-renders history
from scratch: cold KV, no tool_memory splices, no continuation frontier. That
exercises the prompt, not the continuation machinery. This does the loop for
real -- call -> result -> continue, feeding back the model's own tool_call_id --
so ember's exact-token tool replay, post-tool snapshots and continuation
bindings all engage.

Two modes:
  clean   N rounds of ordinary tool use
  abort   same, but one round is CUT MID-STREAM to mimic the production
          "cut off by a network error mid-stream" event. That leaves a partial
          assistant turn whose DSML block may be incomplete; if a truncated
          block gets stored and spliced back, the next turn resumes from
          malformed markup. That is the leading suspect.

Usage: agent_loop.py <port> <tag> <mode:clean|abort> <rounds> <trials>
"""
import http.client
import json
import sys

PORT, TAG, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
ROUNDS, TRIALS = int(sys.argv[4]), int(sys.argv[5])
TEMPLATE = json.load(open("/tmp/capture/req-1.json"))
SCHEMAS = {t["function"]["name"]: (t["function"].get("parameters") or {})
           for t in TEMPLATE["tools"]}
SYS = [m for m in TEMPLATE["messages"] if m.get("role") == "system"]

TYPES = {"string": str, "integer": int, "number": (int, float),
         "array": list, "object": dict, "boolean": bool}

TASKS = [
    "Search the repo for the pattern 'ember_backend_generate' and report what you find.",
    "Now list the files in /tmp with the terminal.",
    "Read /etc/hostname and tell me the value.",
    "Search for files matching 'attention' under /ember.",
    "Run 'uname -a' and summarise the kernel version.",
    "Look up which skills are available.",
    "Search the repo for 'decode_flash'.",
    "Run 'df -h /' and report free space.",
]


def check(schema, args, path=""):
    bad = []
    for r in schema.get("required", []):
        if r not in args:
            bad.append("missing required property %s%s" % (path, r))
    props = schema.get("properties", {})
    for k, v in args.items():
        if k in props:
            want = props[k].get("type")
            if want in TYPES and not isinstance(v, TYPES[want]):
                bad.append("wrong type for %s%s" % (path, k))
    return bad


def post(msgs, cut_after=None):
    """Stream a turn. cut_after=N closes the socket after N chunks (mid-stream)."""
    body = {"model": TEMPLATE["model"], "tools": TEMPLATE["tools"],
            "messages": msgs, "stream": True, "temperature": 0.6,
            "max_tokens": 800}
    conn = http.client.HTTPConnection("127.0.0.1", int(PORT), timeout=900)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    calls, content, nchunk, truncated = {}, "", 0, False
    buf = b""
    while True:
        chunk = resp.read(512)
        if not chunk:
            break
        nchunk += 1
        buf += chunk
        if cut_after is not None and nchunk >= cut_after:
            truncated = True
            conn.close()          # simulate the mid-stream network death
            break
    for line in buf.decode(errors="ignore").splitlines():
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        try:
            d = json.loads(line[6:])
        except Exception:
            continue
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0),
                                        {"id": "", "name": "", "args": ""})
                slot["id"] += tc.get("id") or ""
                fn = tc.get("function", {})
                slot["name"] += fn.get("name") or ""
                slot["args"] += fn.get("arguments") or ""
    try:
        conn.close()
    except Exception:
        pass
    return calls, content, truncated


def trial(t):
    msgs = list(SYS)
    defects = []
    cut_round = (ROUNDS // 2) if MODE == "abort" else None
    for r in range(ROUNDS):
        msgs = msgs + [{"role": "user", "content": TASKS[r % len(TASKS)]}]
        calls, content, truncated = post(
            msgs, cut_after=3 if r == cut_round else None)

        for c in calls.values():
            nm = c["name"]
            if nm not in SCHEMAS:
                defects.append("r%d unknown tool %r" % (r, nm))
                continue
            try:
                args = json.loads(c["args"]) if c["args"].strip() else {}
            except Exception:
                defects.append("r%d %s: arguments not valid JSON: %.50r"
                               % (r, nm, c["args"]))
                continue
            for b in check(SCHEMAS[nm], args):
                defects.append("r%d %s: %s" % (r, nm, b))

        if truncated:
            # Exactly what the gateway does: a length-truncated stub, then the
            # conversation continues from where the stream died.
            msgs = msgs + [{"role": "assistant", "content": content}]
            msgs = msgs + [{"role": "user", "content":
                            "[System: The previous response was cut off by a "
                            "network error mid-stream. Continue.]"}]
            continue

        if not calls:
            msgs = msgs + [{"role": "assistant", "content": content}]
            continue

        asst = {"role": "assistant", "content": content, "tool_calls": []}
        for i, c in enumerate(calls.values()):
            asst["tool_calls"].append(
                {"id": c["id"] or ("call_%d_%d" % (t, r)), "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"]}})
        msgs = msgs + [asst]
        for tc in asst["tool_calls"]:
            msgs = msgs + [{"role": "tool", "tool_call_id": tc["id"],
                            "content": json.dumps({"success": True,
                                                   "result": "ok"})}]
    return defects


tot = bad = 0
for t in range(TRIALS):
    d = trial(t)
    tot += 1
    if d:
        bad += 1
        for x in d:
            print("FAIL [%s/%s] trial%d: %s" % (TAG, MODE, t, x), flush=True)
    else:
        print("ok   [%s/%s] trial%d: %d rounds clean" % (TAG, MODE, t, ROUNDS),
              flush=True)

print("LOOP %s mode=%s trials=%d bad=%d rate=%.4f"
      % (TAG, MODE, tot, bad, bad / tot if tot else 0.0), flush=True)
