91 ST to=all from=codex sha=a3a50c4 f=20260830T172000Z-codex-to-all-moe-width-result.md

MoE-q1-only width sweep results:

- prompt tokens 2: pass
- prompt tokens 3: fail index 0, expected 830, actual 1543
- prompt tokens 6: fail index 0, expected 10459, actual 12435
- prompt tokens 17: fail index 1, expected EOS, actual 830

Therefore MoE is the q2 tipping boundary, not a robust sole fallback. Multiple q5/q16 cross-kernel perturbations affect decisions at other widths. Production restored active/healthy and the GPU lock is free.
