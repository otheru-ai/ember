#!/usr/bin/env python3
"""Prefill AND decode throughput vs context, emitted as JSON lines for plotting.

    max_tokens=256 matches benchmark.py's decode-256 group. At 128 the
    batch-verifier warmup (strict cycles cost ~260ms of verify each, vs ~90ms
    once qualified) is amortised over half as many tokens, which depressed the
    whole spec-on curve by ~20% and made it incomparable to the 37.90 headline.

Generation task is byte-identical at every size (benchmark.py's integer
sequence, ~0.98 acceptance) so the curve shows CONTEXT scaling and not drifting
draft quality -- the confound that invalidated an earlier sweep. Acceptance is
recorded at every point so the reader can check that, not take it on faith.
"""
import json, sys, urllib.error, urllib.request

ENDPOINT, CONFIG = sys.argv[1], sys.argv[2]
TARGETS = [int(x) for x in sys.argv[3].split(",")]
OUT = sys.argv[4]
TOK_PER_WORD = 2.52
TASK = ("Write a very long comma-separated sequence of consecutive positive "
        "integers beginning at 1. Emit only the sequence and continue until the "
        "response token limit; do not conclude early.")

def body(target):
    if target <= 0:
        return f"Marker A. {TASK}"
    w = max(1, int(target / TOK_PER_WORD))
    filler = " ".join(f"W{i}" for i in range(w))
    return f"Background notes (ignore them):\n{filler}\n\nMarker A. {TASK}"

out = open(OUT, "a")
for t in TARGETS:
    payload = {"model": "deepseek-v4-flash",
               "messages": [{"role": "user", "content": body(t)}],
               "reasoning_effort": "none", "temperature": 0,
               "max_tokens": 256, "stream": False}
    req = urllib.request.Request(ENDPOINT, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5400) as f:
            d = json.load(f)
    except urllib.error.HTTPError as e:
        msg = e.read()[:160].decode("utf-8", "replace")
        print(f"   {CONFIG:9s} target={t:<7d} HTTP {e.code}: {msg}")
        continue
    except Exception as e:
        print(f"   {CONFIG:9s} target={t:<7d} FAILED: {e}")
        continue
    u = d.get("usage") or {}
    tm = u.get("timings") or {}
    rec = {"config": CONFIG, "target": t,
           "prompt_tokens": u.get("prompt_tokens"),
           "prefill_tps": tm.get("prefill_tokens_per_sec"),
           "decode_tps": tm.get("decode_tokens_per_sec"),
           "accept": u.get("accept_rate")}
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(f"   {CONFIG:9s} ctx={rec['prompt_tokens']:<7} "
          f"prefill={rec['prefill_tps'] or 0:7.1f}  decode={rec['decode_tps'] or 0:6.2f}  "
          f"accept={rec['accept']}")
