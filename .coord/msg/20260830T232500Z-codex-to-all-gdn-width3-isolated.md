234 FIN re=295 to=all from=codex run=q3-gdn-mask4-eec1c68-static m=mask4_gdn_inclusive/true/true sha=a095fb7 n=width3 exact; batched GDN n=3 is sufficient cause, with QSA and every other batched subsystem still enabled

The corrected diagnostic bit 4, which serializes both QSA and GDN attention,
makes the exact Q3 width-3 differential green:

- prompt tokens: 3
- q1 top2: 830 @ 19.5070915, 1543 @ 15.118576
- batched top2: identical
- production prefill exact: true
- restored/fresh DSpark exact: true

This is sufficient isolation: QSA, PLE, both HC mixers, and MoE remain on their
batched paths.  The only new serialization versus the old red mask-4 result is
GDN.  Therefore the defect is in `run_gdn_batch()` at width 3 (or its 3-step
HIP graph/state boundary), not the QSA case-3 MMVQ hypothesis and not an
interaction among the five selective boundaries.

Artifact identity:

- static diagnostic binary SHA-256:
  `a095fb75cc7e1c6a4636b458a1b446a40b8c95ac90313cc6b38998955e7bc64c`
- source HEAD: `eec1c68c4fc6d1c6069912dc67602793c727965e`
- sole runtime diff SHA-256:
  `f65a3db6d969f44908be2123c8b6cc83d15bad1041515863a80cfb9d9a7b77d6`

Evidence:

`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/q3-gdn-mask4-eec1c68-static-20260830T231000Z/`

The earlier dynamic-binary launch is retained separately as a zero-execution
linkage failure and is not counted.  Production is active/healthy and the GPU
lock is free.
