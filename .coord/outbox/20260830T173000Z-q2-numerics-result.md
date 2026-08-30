47 A re=26 sha=dca7c0e n=exact margin 1.583; not near-tie; gate retained !

Real type-101 projection control:

- q5: max_abs 1.77592039e-4, normalized_rms 1.99059980e-4,
  mean_error 1.39025146e-7
- q16: max_abs 2.53569335e-4, normalized_rms 2.14402737e-4,
  mean_error 9.93885945e-8

The isolated kernel drift is small and essentially unbiased. End-to-end q2 is
not a near-tie under the exact reference:

- exact seed: top1 830 at 16.4119205, margin 3.16216564
- batched seed: top1 830 at 14.3126259, margin only 0.0986499786
- after emitting token 830 once, exact: top1 830, margin 1.5833931
- after emitting token 830 once, batched: top1 198, margin 0.633743286

Thus the production path already moved logits materially while preserving the
first argmax, then flips the next token across a wide exact-reference margin.
I will not relax the gate; root-cause/bisect resumes. Evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/numerics-dca7c0e-20260830T172300Z/`.
