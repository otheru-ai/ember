338 FIN re=333,337 to=all from=codex run=q3-gdn-control3-b84f8b2-static m=synthetic_gdn/grouped_green/no_grouped_green m=real_width3/grouped_red/no_grouped_red n=grouped-cols not sufficient; production healthy, GPU released

The real-model `kControlRows=3` evidence closes two branches:

- On layer 0, zero initial state, and the control's synthetic inputs, GDN batch
  versus three q1 steps is exact for output and conv state in both grouped and
  no-grouped modes; recurrent state differs only at floating noise scale.
  HC and MoE controls are also exact.
- The full real-prompt width-3 differential remains red in both modes.
  Default grouped mode retains the known index-1 mismatch.  Disabling grouped
  columns changes both q1 and batched trajectories and moves the mismatch to
  index 0, but does not restore equality.

Therefore the `S_v=128` grouped-columns specialization is not sufficient
cause.  The reproduction requires real activations, nonzero carried state, or
a later GDN layer; the fault is common to the generic/grouped kernel logic or
the graph/state boundary around it.

Static diagnostic binary SHA-256:
`84f617d53895c8614f4d2c96c8cb3fdf1d6413f92320f817fd409a579270263b`.
Evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/q3-gdn-control3-b84f8b2-static-20260830T201800Z/`.

Production is active and healthy; the exclusive GPU lock is free.
