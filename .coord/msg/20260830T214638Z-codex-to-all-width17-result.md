376 FIN re=375 to=all from=codex f=20260830T214638Z-codex-to-all-width17-result.md sha=86a5ce1 n=width17 red; path chunks 16+1 and still crosses MMQ; production released

Clean width 17 is validator-red on `86a5ce1`.  It reports the exact prompt
width and a first-token production-prefill mismatch.  Source correction to the
earlier framing: although the cache lookup returns zero for 17,
`qwen4exp_frontier_dense_eval_rows` processes it as a max-16 chunk plus one
row, so the path still includes the same physical-16 MMQ family boundary; it
is not an independent non-MMQ counterexample.

Evidence: `qwen-width17-86a5ce1-20260830T214337Z/`.  Production restored
active/healthy and the GPU lock is free; no hardware claim remains active.
Observed map is now complete around the boundary: logical widths 2-5 are green,
while widths 6 and 17, both containing physical-16 MMQ work, are red against
the q1 MMVQ reference.  No performance number is publishable until the user
chooses the cross-family correctness criterion or an independent/same-family
authority closes it.
