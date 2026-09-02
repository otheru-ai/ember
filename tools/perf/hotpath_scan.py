#!/usr/bin/env python3
"""Scan for performance anti-patterns in per-layer / per-token code.

Signal, not proof: every hit needs reading. The point is to enumerate the whole
surface once rather than notice things by accident.
"""
import re, sys, pathlib, collections

PATTERNS = [
    ("ALLOC_IN_LOOP",  r'\bstd::vector<[^>]+>\s+\w+\s*\(', "heap alloc; if in a per-layer/token loop it repeats"),
    ("ZERO_FILL",      r'\bstd::vector<[^>]+>\s+\w+\s*\([^)]*,\s*0(\.0f?)?\s*\)', "allocate AND zero-fill"),
    ("DEVICE_SYNC",    r'(hipDeviceSynchronize|cudaDeviceSynchronize|ggml_backend_synchronize)', "full sync"),
    ("STRING_HOT",     r'(std::string\s+\w+\s*=\s*std::string|\+\s*std::to_string|snprintf)', "string work"),
    ("MAP_LOOKUP",     r'\b(std::map|std::unordered_map)<', "hashed lookup"),
    ("PUSH_NO_RESERVE",r'\.push_back\(', "growth without reserve nearby"),
    ("MEMCPY",         r'\b(std::memcpy|memcpy)\(', "copy; check if a view would do"),
]

def enclosing_loop_depth(lines, idx):
    """Crude: count unclosed for/while braces above within the same function."""
    depth = 0
    brace = 0
    for i in range(idx, max(0, idx - 400), -1):
        l = lines[i]
        brace += l.count('}') - l.count('{')
        if brace < 0:
            if re.search(r'\b(for|while)\s*\(', l):
                depth += 1
            brace = 0
        if re.match(r'^[a-zA-Z_].*\)\s*\{?\s*$', l) and '(' in l and depth:
            break
    return depth

def main(paths):
    hits = collections.defaultdict(list)
    for p in paths:
        try: lines = pathlib.Path(p).read_text().splitlines()
        except Exception: continue
        for i, line in enumerate(lines):
            if line.lstrip().startswith('//'): continue
            for name, pat, _ in PATTERNS:
                if re.search(pat, line):
                    d = enclosing_loop_depth(lines, i)
                    if d > 0 or name in ("DEVICE_SYNC",):
                        hits[name].append((p, i + 1, d, line.strip()[:96]))
    for name, pat, why in PATTERNS:
        h = hits.get(name, [])
        if not h: continue
        print(f"\n=== {name} ({why}): {len(h)} in-loop hits")
        h.sort(key=lambda x: -x[2])
        for p, ln, d, txt in h[:6]:
            print(f"  depth{d} {p.split('/')[-1]}:{ln}  {txt}")
        if len(h) > 6: print(f"  ... {len(h)-6} more")

main(sys.argv[1:])
