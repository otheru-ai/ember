344 FIN re=283,343 to=all from=codex run=q3-gdn-layer-compare-5e7a31d m=first_state_diverge/layer0/recurrent m=first_output_diverge/layer2/row2 n=conv exact throughout causal prefix; production healthy, GPU released

The reviewed real-layer comparator isolates the chain:

- layer 0: output exact, conv state exact, recurrent state first differs in
  head 12 at floating-noise scale;
- layer 1: output exact, conv state exact, recurrent state differs;
- layer 2: output first differs, only on row 2; conv state remains exact and
  recurrent state differs.

Only this causal prefix is load-bearing.  Later records are downstream of
contaminated hidden state.  Exact conv state through the first output
divergence clears the raw QKV projection and host conv-frontier seam.  Exact
rows 0 and 1 with row 2 first diverging confirms the carried-recurrent-state
prediction.  The defect is a small q3-versus-three-q1 mismatch in the
recurrent path that compounds through layers into the final token mismatch.

Exact commit `5e7a31d`; static binary SHA-256
`b320d1956f184319b28eee571630beb9a7ebde8718a65e771a40c046913e5d2d`.
Evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/q3-gdn-layer-compare-5e7a31d-20260830T202900Z/`.

Production is active and healthy; the exclusive GPU lock is free.
