#!/usr/bin/env python3
"""Render the context-scaling chart in docs/assets from a sweep JSONL.

SVG on purpose: no plotting dependency, text in git, and it scales in the
README on any display. Regenerate after a sweep with

    tools/plot_context_scaling.py ctxsweep.jsonl docs/assets/ember-context-scaling.svg

Labels are placed by rule, not at the last data point: the two decode series
converge at depth (14.50 vs 14.86 tok/s at 116k), so end-of-line labels collide.
Series get a legend, and the peak of each curve is called out where it happens.
"""

import json
import math
import sys

W, H = 920, 360
PAD_L, PAD_R, PAD_T, PAD_B = 62, 18, 62, 52
PANEL_GAP = 62
PW = (W - PAD_L - PAD_R - PANEL_GAP) // 2
PH = H - PAD_T - PAD_B

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e8eaed"
SPEC, AR, PREFILL = "#d94f2b", "#8b93a1", "#2b6cb0"


def load(path):
    on, off = {}, {}
    for line in open(path):
        r = json.loads(line)
        (on if r["config"] == "spec-on" else off)[r["prompt_tokens"]] = r
    return on, off


def fmt_depth(n):
    if n >= 1000:
        return f"{n / 1000:.0f}k" if n >= 10000 else f"{n / 1000:.1f}k"
    return str(n)


class Panel:
    def __init__(self, x0, title, vmax, lo, hi):
        self.x0, self.title, self.vmax, self.lo, self.hi = x0, title, vmax, lo, hi
        self.out = []

    def x(self, n):
        return self.x0 + (math.log10(n) - self.lo) / (self.hi - self.lo) * PW

    def y(self, v):
        return PAD_T + PH - (v / self.vmax) * PH

    def frame(self, ticks):
        o = self.out
        o.append(f'<text x="{self.x0}" y="{PAD_T - 34}" font-size="14" '
                 f'font-weight="600" fill="{INK}">{self.title}</text>')
        o.append(f'<text x="{self.x0}" y="{PAD_T - 19}" font-size="10.5" '
                 f'fill="{MUTED}">tokens / second</text>')
        for frac in (0, 0.25, 0.5, 0.75, 1.0):
            v = self.vmax * frac
            yy = self.y(v)
            o.append(f'<line x1="{self.x0}" y1="{yy:.1f}" x2="{self.x0 + PW}" '
                     f'y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
            o.append(f'<text x="{self.x0 - 9}" y="{yy + 3.5:.1f}" font-size="10.5" '
                     f'text-anchor="end" fill="{MUTED}">{v:.0f}</text>')
        for n in ticks:
            o.append(f'<text x="{self.x(n):.1f}" y="{PAD_T + PH + 18}" '
                     f'font-size="10.5" text-anchor="middle" '
                     f'fill="{MUTED}">{fmt_depth(n)}</text>')
        o.append(f'<text x="{self.x0 + PW / 2:.1f}" y="{PAD_T + PH + 38}" '
                 f'font-size="10.5" text-anchor="middle" fill="{MUTED}">'
                 f'prompt depth (tokens)</text>')

    def series(self, pts, colour, width=2.4):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.x(n):.1f},{self.y(v):.1f}"
                     for i, (n, v) in enumerate(pts))
        self.out.append(f'<path d="{d}" fill="none" stroke="{colour}" '
                        f'stroke-width="{width}" stroke-linejoin="round" '
                        f'stroke-linecap="round"/>')
        for n, v in pts:
            self.out.append(f'<circle cx="{self.x(n):.1f}" cy="{self.y(v):.1f}" '
                            f'r="3.2" fill="white" stroke="{colour}" '
                            f'stroke-width="1.8"/>')

    def peak(self, pts, colour):
        n, v = max(pts, key=lambda p: p[1])
        px, py = self.x(n), self.y(v)
        # Keep the callout inside the panel: flip side near the right edge.
        right = px > self.x0 + PW * 0.62
        tx = px - 10 if right else px + 10
        anchor = "end" if right else "start"
        self.out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.6" '
                        f'fill="{colour}"/>')
        self.out.append(f'<text x="{tx:.1f}" y="{py - 13:.1f}" font-size="13" '
                        f'font-weight="700" text-anchor="{anchor}" '
                        f'fill="{colour}">peak {v:.0f} tok/s</text>')
        self.out.append(f'<text x="{tx:.1f}" y="{py - 1:.1f}" font-size="10.5" '
                        f'text-anchor="{anchor}" fill="{MUTED}">'
                        f'at {fmt_depth(n)} depth</text>')

    def legend(self, entries):
        yy = PAD_T + 10
        for label, colour in entries:
            xx = self.x0 + PW - 8
            self.out.append(f'<text x="{xx:.1f}" y="{yy:.1f}" font-size="11" '
                            f'text-anchor="end" fill="{colour}" '
                            f'font-weight="600">{label}</text>')
            self.out.append(f'<line x1="{xx - 92:.1f}" y1="{yy - 4:.1f}" '
                            f'x2="{xx - 74:.1f}" y2="{yy - 4:.1f}" '
                            f'stroke="{colour}" stroke-width="2.4"/>')
            yy += 16


def main():
    src, dst = sys.argv[1], sys.argv[2]
    on, off = load(src)
    xs = sorted(on)
    lo, hi = math.log10(xs[0]), math.log10(xs[-1])
    # Thin the depth ticks: every point crowds the log axis at the low end.
    ticks = [n for i, n in enumerate(xs) if i != 1] if len(xs) > 6 else xs

    pre = [(n, on[n]["prefill_tps"]) for n in xs]
    dec_on = [(n, on[n]["decode_tps"]) for n in xs]
    dec_off = [(n, off[n]["decode_tps"]) for n in xs if n in off]

    def ceil_to(v, step):
        return math.ceil(v / step) * step

    p1 = Panel(PAD_L, "Prefill", ceil_to(max(v for _, v in pre), 100), lo, hi)
    p1.frame(ticks)
    p1.series(pre, PREFILL)
    p1.peak(pre, PREFILL)

    dmax = max([v for _, v in dec_on] + [v for _, v in dec_off])
    p2 = Panel(PAD_L + PW + PANEL_GAP, "Decode", ceil_to(dmax, 10), lo, hi)
    p2.frame(ticks)
    if dec_off:
        p2.series(dec_off, AR, width=2.0)
    p2.series(dec_on, SPEC)
    p2.peak(dec_on, SPEC)
    p2.legend([("speculative", SPEC)] + ([("autoregressive", AR)] if dec_off else []))

    body = [f'<rect width="{W}" height="{H}" fill="white"/>'] + p1.out + p2.out
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,'
           f'Segoe UI,Roboto,Helvetica,Arial,sans-serif">\n  '
           + "\n  ".join(body) + "\n</svg>\n")
    with open(dst, "w") as fh:
        fh.write(svg)
    print(f"wrote {dst} ({len(svg)} bytes, {len(xs)} depths)")


if __name__ == "__main__":
    main()
