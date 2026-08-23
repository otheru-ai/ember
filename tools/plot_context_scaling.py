#!/usr/bin/env python3
"""Render the context-scaling chart in docs/assets from a sweep JSONL.

SVG on purpose: no plotting dependency, text in git, and it scales in the
README on any display. Regenerate after a sweep with

    tools/plot_context_scaling.py ctxsweep.jsonl docs/assets/ember-context-scaling.svg
"""

import json
import math
import sys

W, H = 880, 300
PAD_L, PAD_R, PAD_T, PAD_B = 58, 14, 30, 40
PANEL_GAP = 46
PW = (W - PAD_L * 2 - PANEL_GAP) // 2
PH = H - PAD_T - PAD_B

INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
SPEC = "#d94f2b"
AR = "#6b7280"
PREFILL = "#2b6cb0"


def load(path):
    on, off = {}, {}
    for line in open(path):
        r = json.loads(line)
        (on if r["config"] == "spec-on" else off)[r["prompt_tokens"]] = r
    return on, off


def xpos(n, lo, hi, x0):
    return x0 + (math.log10(n) - lo) / (hi - lo) * PW


def ypos(v, vmax, y0):
    return y0 + PH - (v / vmax) * PH


def panel(x0, title, series, vmax, lo, hi, xs, unit):
    out = [f'<text x="{x0}" y="{PAD_T - 12}" font-size="13" font-weight="600" '
           f'fill="{INK}">{title}</text>']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        v = vmax * frac
        y = ypos(v, vmax, PAD_T)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + PW}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" font-size="10" '
                   f'text-anchor="end" fill="{MUTED}">{v:.0f}</text>')
    for n in xs:
        x = xpos(n, lo, hi, x0)
        label = f"{n // 1000}k" if n >= 1000 else str(n)
        out.append(f'<text x="{x:.1f}" y="{PAD_T + PH + 16}" font-size="10" '
                   f'text-anchor="middle" fill="{MUTED}">{label}</text>')
    out.append(f'<text x="{x0 + PW / 2:.1f}" y="{PAD_T + PH + 33}" font-size="10" '
               f'text-anchor="middle" fill="{MUTED}">prompt tokens</text>')
    for label, pts, colour in series:
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{xpos(n, lo, hi, x0):.1f},{ypos(v, vmax, PAD_T):.1f}"
            for i, (n, v) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2.2" '
                   f'stroke-linejoin="round"/>')
        for n, v in pts:
            out.append(f'<circle cx="{xpos(n, lo, hi, x0):.1f}" '
                       f'cy="{ypos(v, vmax, PAD_T):.1f}" r="3" fill="{colour}"/>')
        n, v = pts[-1]
        out.append(f'<text x="{xpos(n, lo, hi, x0) - 4:.1f}" '
                   f'y="{ypos(v, vmax, PAD_T) - 9:.1f}" font-size="10" '
                   f'text-anchor="end" fill="{colour}" font-weight="600">{label}</text>')
    out.append(f'<text x="{x0 - 8}" y="{PAD_T - 12}" font-size="10" '
               f'text-anchor="end" fill="{MUTED}">{unit}</text>')
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    on, off = load(src)
    xs = sorted(on)
    lo, hi = math.log10(xs[0]), math.log10(xs[-1])

    pre = [(n, on[n]["prefill_tps"]) for n in xs]
    dec_on = [(n, on[n]["decode_tps"]) for n in xs]
    dec_off = [(n, off[n]["decode_tps"]) for n in xs if n in off]

    body = [f'<rect width="{W}" height="{H}" fill="white"/>']
    body += panel(PAD_L, "Prefill", [("prefill", pre, PREFILL)],
                  400, lo, hi, xs, "tok/s")
    body += panel(PAD_L + PW + PANEL_GAP, "Decode",
                  [("speculative", dec_on, SPEC), ("autoregressive", dec_off, AR)],
                  40, lo, hi, xs, "tok/s")

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,'
           f'Segoe UI,Roboto,sans-serif">\n  ' + "\n  ".join(body) + "\n</svg>\n")
    with open(dst, "w") as fh:
        fh.write(svg)
    print(f"wrote {dst} ({len(svg)} bytes, {len(xs)} points)")


if __name__ == "__main__":
    main()
