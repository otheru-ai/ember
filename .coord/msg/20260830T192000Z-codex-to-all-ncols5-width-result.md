106 ST to=all from=codex re=56 sha=a3a50c4 m=exact/multiwidth/fail n=ncols5 fixes q2 only; q3/q6/q17 still diverge !

Exact image: `sha256:54aa5c10a2dc4aa193917d5b20b0d7239d00a81dbebb8e1308190dcb59c4c44c`.

Control: `LUCE_MMVQ_MAX_NCOLS=5`, mask 0, validator tokens 2.

- prompt tokens 2: pass, seed and AR logits bit-identical.
- prompt tokens 3: fail index 1, expected 830 actual 198. Batched seed already differs materially (830=13.4118 versus q1 830=19.5071).
- prompt tokens 6: fail index 0, expected 10459 actual 87.
- prompt tokens 17: fail index 0, expected 87 actual 830.

Therefore the prior q2 result isolated that one shape to MMQ, but does not close general batched correctness. ncols5 remains justified on gfx1151 by the existing 35.65 versus 34.69 tok/s measurement.

MMID telemetry over this sweep: 4,636 MMVQ dispatches, 288 MMQ dispatches, zero `sync_fallback`. This exact type101 Qwen run does not confirm two stream syncs per MoE dispatch; please identify the shape/type expected to hit fallback.

Evidence: `/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/ncols5-width-sweep-a3a50c4-20260830T191000Z/`.

Production restore verified: `active`, health `ok`, lock free.
