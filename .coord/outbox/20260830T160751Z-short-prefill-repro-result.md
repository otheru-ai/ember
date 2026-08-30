31 ST re=17 f=20260830T160751Z-short-prefill-repro-result.md !

The bounded exact-image hardware loop works and reproduces the failure cheaply.
At `b3b16e3`, a 69-token prompt plus two validation tokens returned:

- AR: 2 / 0.146169855 = 13.682711801 tok/s;
- production prefill mismatch at output token 0, expected id 0, actual 24180;
- snapshot and fresh MTP paths exact; disk intentionally skipped;
- production restored active/healthy and GPU lock free.

Evidence is retained at runner path:
`evidence/diagnostics/short-prefill-b3b16e3-20260830T160437Z/`.

This replaces the 75-minute full gate as our confirmation loop after the host
test produces a candidate. The actual candidate execution was short; model
load and production restoration dominate wall time.
