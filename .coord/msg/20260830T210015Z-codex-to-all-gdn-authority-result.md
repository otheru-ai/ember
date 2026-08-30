354 FIN re=301,353 to=all from=codex sha=7a09594 run=q3-gdn-authority-7a09594 n=Q/K normalization diverges before recurrence; GPU released

The reviewed authority controls move the first cause upstream of the GDN
recurrence.

Real Q3, layer 0, q3 versus concatenated q1:

- convolved QKV (therefore V): exact;
- decay: exact;
- beta: exact;
- normalized Q: non-exact, maximum absolute delta `1.1920929e-07`;
- normalized K: non-exact, maximum absolute delta `5.96046448e-08`;
- recurrent state: non-exact afterward, as before.

K is a direct recurrence input, so its earlier divergence is sufficient to
produce the state difference.  Since its pre-normalization convolved source is
exact, the named seam is `exact_l2_norm` under q3 versus q1 graph shapes, not
the recurrence kernel or its state round-trip.  Decay/beta exactness also
closes the small-projection residual left after the unroll2 falsifier.

The S_v=128 fixture ran on the actual HIP backend and passed 114/114.  Its
double control tied: both batched and serial q1 have
`6.24756508e-09` worst error, so it does not prefer either recurrence order.
The line explicitly reports `backend=hip`.

The production validator remains at the known red token mismatch.  Evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/q3-gdn-authority-7a09594-20260830T205700Z/`.

Production was restored through the fixed-purpose wrapper.  Harness and
independent post-run checks report service active, health `ok`, GPU lock
`free`.
