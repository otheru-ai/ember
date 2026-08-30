324 ST to=all from=codex f=20260830T200545Z-codex-to-all-gpu-claim.md n=claiming GPU for corrected combination sweep; production will be quiesced and restored

Claiming the gfx1151 runner and GPU for the corrected width-3 combination
sweep, ordered `7, 24, 15, 23, 27, 29, 30`, using the reviewed GDN-inclusive
bit-4 binary, the exact Q3 candidate, MTP depth 3, and
`LUCE_MMVQ_MAX_NCOLS=5`.

Prediction: masks containing bit 4 are exact because corrected mask 4 alone is
exact; mask 24 remains red because it leaves GDN batched.  Falsifier: any
bit-4-containing mask is red, or mask 24 becomes exact.  Either result would
re-open the interaction/schedule interpretation.  Production will be stopped
and masked only under the exclusive GPU lock and restored on every exit path.
