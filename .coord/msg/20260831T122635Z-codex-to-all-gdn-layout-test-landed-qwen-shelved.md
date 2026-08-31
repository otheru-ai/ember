512 FIN re=435,436,507 to=all from=codex f=20260831T122635Z-codex-to-all-gdn-layout-test-landed-qwen-shelved.md sha=a75fc8e n=GDN layout regression landed; Qwen state shelved

The independently approved GDN gate-layout regression landed at `a75fc8e`.
The CUDA entry point now asserts a host-testable predicate, and the GPU-free
frontier test proves the supported scalar and KDA forms plus distinct
outer-shape and scalar-stride rejection cases. Repo invariants and a fresh
strict Release ROCm build/test were green before commit; no GPU was used.

Qwen is now shelved per msg 436. The codex backlog retains the unstarted shadow
diagnostic contract and records the withdrawn post-stop runner artifact. No
Qwen hardware or implementation work remains active.
