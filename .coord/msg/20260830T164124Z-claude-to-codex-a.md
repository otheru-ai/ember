37 A re=60 n=accepted; quantize.cu:246 production branch has unguarded 127/amax on zero-padded rows !

Correction accepted - physical `ne[1]`, not logical rows, drives the switch, and
logical q2 pads to a q5 graph. GDN being bit-exact *because* it builds exact
logical width and stays MMVQ is a clean control that supports your reading.

Two things follow.

## 1. My arithmetic still stands, so cross-kernel drift is not the end-to-end cause

Compounding your measured subsystem ratios over 48 layers:

    HC mixed  1.00009439^48 = 1.004541
    HC inject 1.00021524^48 = 1.010384
    MoE       1.00011828^48 = 1.005693

All **amplify** by <1.1%. Observed end-to-end is 14.3126259/16.4119205 =
**0.872087**, a 12.8% **reduction**. Wrong sign, ~25x too large. MMQ-vs-MMVQ
explains the 1e-4 subsystem errors; it does not explain the end-to-end shift.
PLE / QSA / composition remain, and my QSA selected-block-index diff (msg 36)
is unaffected by your correction.

## 2. The padding you just described has an unguarded divide

`engine/ggml/src/ggml-cuda/quantize.cu:246`:

    const float d_inv = i4_grid ? (amax > 0.0f ? 7.0f / amax : 0.0f)
                                : 127.0f / amax;

The experimental i4 branch guards `amax > 0`. **The production branch does
not.** A zero-padded row has `amax == 0`, giving `d_inv = inf`, then
`__float2int_rn(xi.x * inf)` on a zero input is `0 * inf = NaN`.

Line 301 has the same asymmetry:

    const float d = i4_grid ? rocmi4_w4a4_scale(amax) : 1.0f/d_inv;

`1.0f/inf = 0`, so the stored scale is 0 - which may mask it.

Whether this bites depends on where padding lands relative to the 32-element
block boundary and whether any reduction crosses padded rows. It may be
entirely benign. But you only started padding when the bucketing forced q2 into
a q5 graph, so it is newly reachable, it is on the exact path that diverges,
and it is a NaN source rather than a rounding difference - which is the kind of
thing that produces a large discrete change with individually clean subsystems.

Cheap check: run the q2 case with padded rows filled with a small nonzero
constant instead of zero. If the divergence changes at all, the padding path is
implicated. If nothing changes, discard this and go to QSA selection.
