#!/usr/bin/env python3
"""De-confound the contamination result, and test the normalisation directly.

The earlier pruning test (0/5 asis vs 5/5 pruned) removed WHOLE TURNS: 36
messages down to 12. So it varied two things at once -- the parameterless shape
AND the length/complexity of the context. It cannot distinguish them.

This isolates the shape. Every message is kept; only the ARGUMENTS of
empty-argument calls change, from {} to a real schema-declared property. If the
failure is caused by parameterless <invoke> blocks in history, normalised should
recover; if it is caused by the long degraded context, normalised should still
fail and the proposed sanitize_api_messages change is not worth building.

  asis        untouched
  normalised  {} -> {"mode": "browse"} on session_search / skills_list, same
              message count, same everything else
  pruned      the original whole-turn removal, for reference

Usage: normalise_test.py <reps>
"""
import copy
import json
import sys
import urllib.request

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
URL = "http://127.0.0.1:8000/v1/chat/completions"
BASE = json.load(open("/tmp/capture/live-16-req.json"))

# Values that are inert at the tool layer: the registry handler reads only
# named keys, so session_search() never receives "mode".
NOOP_ARGS = {"session_search": {"mode": "browse"},
             "skills_list": {"mode": "list"}}


def parsed_args(tc):
    a = tc.get("function", {}).get("arguments")
    try:
        return json.loads(a) if a else {}
    except Exception:
        return None


def normalise(msgs):
    out = []
    changed = 0
    for m in msgs:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = copy.deepcopy(tc)
                p = parsed_args(tc)
                nm = tc.get("function", {}).get("name")
                if p == {} and nm in NOOP_ARGS:
                    tc["function"]["arguments"] = json.dumps(NOOP_ARGS[nm])
                    changed += 1
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out, changed


def prune(msgs):
    drop, out = set(), []
    for m in msgs:
        if m.get("role") == "assistant" and (m.get("tool_calls") or []):
            if all(parsed_args(tc) in ({}, None) for tc in m["tool_calls"]):
                for tc in m["tool_calls"]:
                    if tc.get("id"):
                        drop.add(tc["id"])
                continue
        if m.get("role") == "tool" and m.get("tool_call_id") in drop:
            continue
        out.append(m)
    return out


def run(msgs):
    body = dict(BASE)
    body["messages"] = msgs
    body["stream"] = True
    body["max_tokens"] = 1500
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=900).read().decode()
    calls, content = {}, []
    for line in raw.splitlines():
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        try:
            d = json.loads(line[6:])
        except Exception:
            continue
        for ch in d.get("choices", []):
            de = ch.get("delta", {})
            if de.get("content"):
                content.append(de["content"])
            for tc in de.get("tool_calls") or []:
                s = calls.setdefault(tc.get("index", 0), {"a": ""})
                s["a"] += (tc.get("function", {}) or {}).get("arguments") or ""
    valid = 0
    for v in calls.values():
        try:
            if json.loads(v["a"] or "{}"):
                valid += 1
        except Exception:
            pass
    dropped = (not calls) and (not "".join(content).strip())
    return len(calls), valid, dropped


norm, nchanged = normalise(copy.deepcopy(BASE["messages"]))
ARMS = {
    "asis": BASE["messages"],
    "normalised": norm,
    "pruned": prune(copy.deepcopy(BASE["messages"])),
}
print("normalised: rewrote %d empty-argument calls, message count %d -> %d"
      % (nchanged, len(BASE["messages"]), len(norm)), flush=True)
for name, msgs in ARMS.items():
    print("ARM %-11s messages=%d" % (name, len(msgs)), flush=True)

for name, msgs in ARMS.items():
    ok = dropped = 0
    for i in range(REPS):
        try:
            n, v, d = run(msgs)
        except Exception as e:
            print("  %-11s rep%d ERROR %s" % (name, i, e), flush=True)
            continue
        if d:
            dropped += 1
        if n and v == n:
            ok += 1
        print("  %-11s rep%d calls=%d valid=%d dropped=%s" % (name, i, n, v, d),
              flush=True)
    print("NORM %-11s reps=%d all_valid=%d dropped=%d" % (name, REPS, ok, dropped),
          flush=True)
