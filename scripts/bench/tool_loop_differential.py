#!/usr/bin/env python3
"""Does speculation change what the model decides after a tool result?

    tool_loop_differential.py --endpoint URL --out results.jsonl [--loops N]

Speculation is withheld from every turn that follows a tool result. The reason
is recorded at src/server/main.c: speculative verification can change a
near-tied token, and was once seen to re-emit an identical successful
write_file call -- the model repeating an action it had already completed.

That was never quantified, and it is expensive: on a live agent workload the
rule covered 91 of 167 turns and roughly two thirds of all decode time. A
37,869-token differential against autoregressive found no divergence anywhere,
but it could not reach this case: tool-result turns are forced to AR, so the
comparison would have been AR against AR.

EMBER_TOOL_RESULT_AR=0 lifts the rule. This driver runs the same tool loops
against a server with it lifted and a server without, and compares what the
model decided at each step. Text equality is necessary but not sufficient: the
failure has a specific signature, so the emitted tool calls are compared as
structured values, and a repeated call to an already-succeeded tool is reported
whether or not the surrounding prose matched.

Run it twice -- once per server -- then diff with --compare.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# Deliberately mutating tools: re-issuing a read is harmless, re-issuing a write
# is the failure being hunted.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a path. Overwrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the entries of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

# Each scenario ends with a tool result, so the next decision is exactly the
# turn the rule governs.
SCENARIOS = [
    ("write-then-next",
     "Create a file at /tmp/report.md containing a one-line summary, then tell "
     "me what to do next.",
     "write_file", {"path": "/tmp/report.md", "content": "summary\\n"},
     '{"ok": true, "bytes_written": 8}'),
    ("write-then-verify",
     "Write /tmp/config.yaml with a single key, then confirm the result.",
     "write_file", {"path": "/tmp/config.yaml", "content": "key: value\\n"},
     '{"ok": true, "bytes_written": 11}'),
    ("list-then-act",
     "List /tmp and then decide whether anything needs writing.",
     "list_dir", {"path": "/tmp"},
     '{"entries": ["report.md", "config.yaml"]}'),
    ("write-failed-then-retry",
     "Write /tmp/locked.txt with the word hello.",
     "write_file", {"path": "/tmp/locked.txt", "content": "hello\\n"},
     '{"ok": false, "error": "permission denied"}'),
]


def call(endpoint: str, messages: list, max_tokens: int) -> dict:
    body = {
        "model": "deepseek-v4-flash",
        "reasoning_effort": "none",
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        endpoint, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1200) as fh:
        return json.load(fh)


def tool_calls_of(message: dict) -> list:
    """Normalised tool calls: name and arguments, ids dropped.

    Ids differ between runs by construction and say nothing about the decision.
    """
    out = []
    for c in message.get("tool_calls") or []:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        out.append({"name": fn.get("name"), "arguments": args})
    return out


def run(endpoint: str, out_path: str, loops: int, max_tokens: int) -> None:
    with open(out_path, "w") as fh:
        for rep in range(loops):
            for name, ask, tool, args, result in SCENARIOS:
                messages = [
                    {"role": "user", "content": ask},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{
                         "id": "call_1", "type": "function",
                         "function": {"name": tool,
                                      "arguments": json.dumps(args)}}]},
                    {"role": "tool", "tool_call_id": "call_1",
                     "content": result},
                ]
                try:
                    d = call(endpoint, messages, max_tokens)
                except Exception as exc:                    # noqa: BLE001
                    fh.write(json.dumps(
                        {"scenario": name, "rep": rep, "error": str(exc)}) + "\n")
                    fh.flush()
                    print("  %-24s rep=%d ERROR %s" % (name, rep, exc), flush=True)
                    continue
                msg = d["choices"][0]["message"]
                usage = d["usage"]
                calls = tool_calls_of(msg)
                rec = {
                    "scenario": name,
                    "rep": rep,
                    "content": msg.get("content") or "",
                    "tool_calls": calls,
                    # The signature of the original incident: the model calls
                    # the same mutating tool again with the same arguments.
                    "repeated_prior_call": any(
                        c["name"] == tool and c["arguments"] == args
                        for c in calls),
                    "spec_ran": usage["backend"]["spec_ran"],
                    "accept_rate": usage.get("accept_rate"),
                    "completion_tokens": usage["completion_tokens"],
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print("  %-24s rep=%d spec=%-5s calls=%d repeat=%-5s tok=%d"
                      % (name, rep, rec["spec_ran"], len(calls),
                         rec["repeated_prior_call"], rec["completion_tokens"]),
                      flush=True)


def compare(spec_path: str, ar_path: str) -> int:
    def load(path):
        rows = {}
        for line in open(path):
            r = json.loads(line)
            if "error" in r:
                continue
            rows[(r["scenario"], r["rep"])] = r
        return rows

    spec, ar = load(spec_path), load(ar_path)
    keys = sorted(set(spec) & set(ar))
    spec_ran = [k for k in keys if spec[k]["spec_ran"]]

    text_diff = [k for k in spec_ran if spec[k]["content"] != ar[k]["content"]]
    call_diff = [k for k in spec_ran
                 if spec[k]["tool_calls"] != ar[k]["tool_calls"]]
    repeat_spec = [k for k in spec_ran if spec[k]["repeated_prior_call"]]
    repeat_ar = [k for k in spec_ran if ar[k]["repeated_prior_call"]]
    tokens = sum(spec[k]["completion_tokens"] for k in spec_ran)

    print("  paired turns:                    %d" % len(keys))
    print("  of those, speculation ran:       %d" % len(spec_ran))
    print("  tokens decided under speculation:%d" % tokens)
    print()
    print("  differing prose:                 %d" % len(text_diff))
    print("  differing tool calls:            %d" % len(call_diff))
    print("  repeated a succeeded call (spec):%d" % len(repeat_spec))
    print("  repeated a succeeded call (ar):  %d" % len(repeat_ar))

    if not spec_ran:
        print()
        print("  NOTHING SPECULATED. The rule was not lifted: check that the"
              " server ran with EMBER_TOOL_RESULT_AR=0.")
        return 2

    for k in call_diff:
        print()
        print("  tool-call divergence in %s rep=%d:" % k)
        print("    spec -> %s" % json.dumps(spec[k]["tool_calls"]))
        print("    ar   -> %s" % json.dumps(ar[k]["tool_calls"]))

    # A repeat that only speculation produces is the original incident.
    only_spec = set(repeat_spec) - set(repeat_ar)
    if only_spec:
        print()
        print("  REPRODUCED: speculation re-emitted an already-successful call"
              " where autoregressive did not, in %d turn(s):" % len(only_spec))
        for k in sorted(only_spec):
            print("    %s rep=%d" % k)
        return 1
    if not call_diff and not text_diff:
        print()
        print("  No divergence: speculation and autoregressive decided"
              " identically on every paired turn.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint")
    ap.add_argument("--out")
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--compare", nargs=2, metavar=("SPEC", "AR"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.endpoint or not args.out:
        ap.error("--endpoint and --out are required unless --compare is given")
    run(args.endpoint, args.out, args.loops, args.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
