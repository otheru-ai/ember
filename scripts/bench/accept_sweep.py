import json, urllib.request, sys, time

EP = "http://127.0.0.1:8000/v1/chat/completions"
# Prompts chosen to span predictability of the CONTINUATION, which is what
# acceptance actually depends on -- not properties of the prompt text.
PROMPTS = [
 ("count",      "Write a very long comma-separated sequence of consecutive positive integers beginning at 1. Emit only the sequence and continue until the response token limit; do not conclude early."),
 ("repeat",     "Repeat the exact sentence The quick brown fox jumps over the lazy dog. over and over until the token limit. Emit nothing else."),
 ("alphabet",   "List the letters of the English alphabet in order, then repeat the list again and again until the token limit. Emit only letters and commas."),
 ("multiples",  "Write the multiples of 7 in ascending order starting at 7, comma separated, until the token limit. Emit only the numbers."),
 ("json",       "Emit a JSON array of 200 objects, each exactly {\"id\": N, \"name\": \"item-N\", \"active\": true} with N counting from 1. Emit only JSON."),
 ("code",       "Write a Python function for each of these: add, subtract, multiply, divide, modulo, power. Use identical docstring and body style for each. Emit only code."),
 ("factual",    "List the planets of the solar system in order from the Sun, then their approximate diameters, then their orbital periods. Use short lines."),
 ("prose",      "Explain why deterministic validation matters for an inference server. Use at least eight sentences."),
 ("essay",      "Write a detailed original essay on the philosophy of measurement in computer systems engineering. Be specific and avoid cliches."),
 ("creative",   "Invent an original short story about a lighthouse keeper who discovers something unexpected. Use vivid, unusual language."),
]

def run(label, prompt, max_tokens=256):
    payload = {"model":"deepseek-v4-flash","messages":[{"role":"user","content":prompt}],
               "reasoning_effort":"none","temperature":0,"max_tokens":max_tokens,"stream":False}
    req = urllib.request.Request(EP, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        b = json.load(r)
    u = b.get("usage") or {}; t = u.get("timings") or {}
    return {"label":label, "prompt_tokens":u.get("prompt_tokens"),
            "completion_tokens":u.get("completion_tokens"),
            "decode_tps":t.get("decode_tokens_per_sec"),
            "accept_rate":b.get("accept_rate")}

out = sys.argv[1]
with open(out,"w") as f:
    for label, p in PROMPTS:
        try:
            rec = run(label, p)
        except Exception as e:
            rec = {"label":label, "error":str(e)}
        f.write(json.dumps(rec)+"\n"); f.flush()
        print(json.dumps(rec), flush=True)
