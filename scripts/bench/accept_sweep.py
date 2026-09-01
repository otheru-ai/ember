import json, os, struct, sys, urllib.request, time, zlib

# Endpoint is an argument, not a constant: the bundle orchestrator serves on a
# private port, and a hardcoded one silently produced a file of connection
# errors that still assembled into a bundle.
EP = sys.argv[1] if len(sys.argv) > 2 else "http://127.0.0.1:8000/v1/chat/completions"
# The model name is not a constant either. The vision candidate serves a
# different id, and a hardcoded one produces a file of 404s that still
# assembles into a bundle -- the same failure the endpoint comment describes.
MODEL = os.environ.get("EMBER_BENCH_MODEL", "deepseek-v4-flash")
# Vision workloads are OPT-IN. Text-only releases have no vision path, so
# running them unconditionally would turn every historical release into a file
# of errors. build_perf_site_data.py takes the UNION of per-release workloads,
# so a release that omits them simply has no vision rows -- the dashboard needs
# no change and old bundles stay valid.
VISION = os.environ.get("EMBER_BENCH_VISION", "") == "1"
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

def _png(width, height):
    """A deterministic PNG, generated rather than committed.

    Ember accepts PNG by SIGNATURE, not by the declared media type, so a JPEG
    here would be rejected outright. Generating it keeps the harness
    self-contained: no binary in git, no dependency on /srv/models, and the
    same bytes on every run, which is what makes a perf number comparable
    across releases.

    The content is deliberately structured rather than noise: a flat field
    compresses to almost nothing and would understate both transfer and the
    tower's work.
    """
    raw = bytearray()
    for y in range(height):
        raw.append(0)                       # filter type 0 for every row
        for x in range(width):
            raw += bytes((
                (x * 7 + y * 3) & 0xFF,
                (x ^ y) & 0xFF,
                ((x >> 3) * 11 + (y >> 3) * 5) & 0xFF,
            ))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


def _image_url():
    import base64
    # 1024x680 matches the natural gate corpus, so tower cost here is
    # representative of a real request rather than a toy.
    return "data:image/png;base64," + base64.b64encode(_png(1024, 680)).decode()


# Vision workloads. Two, not one, because they load different parts of the
# path: image_short is dominated by tower encode plus prefill and says almost
# nothing about decode, while image_long amortises the tower over a long
# continuation. Reporting only one would hide whichever half regressed.
VISION_PROMPTS = [
 ("image_short", "What is in this image? Answer with as few words as possible.", 32),
 ("image_long",  "Describe this image in detail, then continue describing until the token limit.", 256),
]


def run(label, prompt, max_tokens=256, image=None):
    content = prompt if image is None else [
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": prompt},
    ]
    payload = {"model":MODEL,"messages":[{"role":"user","content":content}],
               "reasoning_effort":"none","temperature":0,"max_tokens":max_tokens,"stream":False}
    req = urllib.request.Request(EP, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        b = json.load(r)
    u = b.get("usage") or {}; t = u.get("timings") or {}
    backend = u.get("backend") or {}
    return {"label":label, "prompt_tokens":u.get("prompt_tokens"),
            "completion_tokens":u.get("completion_tokens"),
            "decode_tps":t.get("decode_tokens_per_sec"),
            "prefill_tps":t.get("prefill_tokens_per_sec"),
            "accept_rate":u.get("accept_rate"),
            "spec_ran":backend.get("spec_ran"),
            "spec_cycles":t.get("spec_cycles")}

out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
with open(out,"w") as f:
    work = [(label, p, 256, None) for label, p in PROMPTS]
    if VISION:
        url = _image_url()
        work += [(label, p, mt, url) for label, p, mt in VISION_PROMPTS]
    for label, p, mt, img in work:
        try:
            rec = run(label, p, mt, img)
        except Exception as e:
            rec = {"label":label, "error":str(e)}
        f.write(json.dumps(rec)+"\n"); f.flush()
        print(json.dumps(rec), flush=True)
