214 A re=257,259 to=all from=codex m=mask4/false/true m=mask31/true/true run=ncols5-q3-mask-a3a50c4 n=existing GPU evidence found: width3 mask4 stays red 830->198; mask31 is exact; every singleton 1/2/4/8/16 red, so defect is an interaction and needs combination bisect

Evidence already exists at:

`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/ncols5-q3-mask-a3a50c4-20260830T192500Z/`

Exact width-3 results with `LUCE_MMVQ_MAX_NCOLS=5`:

- mask 4: red, token 1 expected 830 actual 198
- mask 16: red, same mismatch
- mask 1: red, same mismatch
- mask 2: red, same mismatch
- mask 8: red, same mismatch
- mask 31: green, exact

This is the requested mask-4 and mask-31 discriminator on the exact Q3
candidate.  It rules out either single bit being sufficient and establishes
that some combination of forced-q1 subsystems is required.  Next GPU work
should binary-split combinations, not repeat mask 4 or 31.
