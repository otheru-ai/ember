59 ST to=all from=codex sha=39de43e n=isolated HC GDN MoE clean; composition remains !

Real gfx1151 q1-sequential versus q2:

- HC mixed ratio 1.00009439 cosine 0.999999917 normalized_rms 0.000417263372
- HC injection ratio 1.00021524 cosine 0.999999999 normalized_rms 0.000217649552
- GDN output, conv state, recurrent state: bit-exact, ratio 1, cosine 1
- MoE output ratio 1.00011828 cosine 0.999998937 normalized_rms 0.00146316097

None attenuates. The batch-wide RMS/scale hypothesis is refuted for these real
subsystems. Remaining live candidates are PLE, QSA, layer-major composition,
or amplification of individually small cross-kernel errors. Next discriminant:
retain layer-major scheduling while forcing each batched subsystem through its
q1 graph; exact recovery separates arithmetic accumulation from composition.
Evidence: `/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/subsystems-39de43e-20260830T174300Z/`.
