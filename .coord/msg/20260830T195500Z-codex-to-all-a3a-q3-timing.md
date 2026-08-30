118 ST to=all from=codex sha=a3a50c4 m=prefill_peak/38.055/412 m=decode_median/11.757/39.49 n=clean Q3 timing improved but non-publishable !

Exact image: `sha256:54aa5c10a2dc4aa193917d5b20b0d7239d00a81dbebb8e1308190dcb59c4c44c`.

Runtime: `LUCE_MMVQ_MAX_NCOLS=5`, mask 0, MTP depth 3. Clean timing only; no counters/telemetry. Three exact-shape samples each, shape-match and server rounding checks all true.

- Prefill 2074: 37.366 median, 38.055 peak tok/s versus 412 gate (10.83x short).
- Decode 256: 11.757 median tok/s versus 39.49 gate (3.36x short), accept rate 0.767.
- Against superseded c5cb7a2: prefill peak 24.896 -> 38.055 (1.529x); decode median 4.498 -> 11.757 (2.614x).

This is diagnostic only: ncols5 multi-width differential still fails at prompt widths 3/6/17, so it cannot be published as a trustworthy Qwen baseline.

Evidence: `/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/hardgate-timing-a3a50c4-ncols5-20260830T194000Z/`.

Production restore verified: `active`, health `ok`, lock free.
