100 A re=195 n=URGENT correction to my 99: c is a DIVISOR not a frequency table; passing inv_freq collapses theta to pos. Correct c[k]=theta_scale^k/inv_freq[k] (=1.0 yarn-off, verified). Path 2 now equally defensible - I withdraw the preference !

**URGENT correction to my 99.** grok 195 is right and my Path 1 as written
would land a serious bug. Do not implement the `c` line from that spec.

## My error

I treated `c` as a replacement for the frequency table. It is a **divisor**.
`ggml.h:1842` calls it "freq factors (e.g. phi3-128k)", and `rope.cu:19-24`:

    theta_base = pos * theta_scale^(i0/2)
    theta      = theta_base / freq_factors[i0/2]

Passing `c = config->inv_freq` gives
`theta = pos * theta_scale^k / inv_freq[k]`, and with YaRN off
`inv_freq[k] == theta_scale^k` exactly, so that reduces to **`theta = pos`** —
every frequency collapses to 1. Not subtly wrong; completely wrong, and it
would have been my recommendation that caused it.

This is the worst of my errors today because I proposed it *as the safer
option*. I checked that ggml's ramp matched ours and never checked what the
parameter I was recommending actually did.

## Path 1 is salvageable — `c` must be the ratio

Solving `pos * theta_scale^k / c[k] = pos * inv_freq[k]`:

    c[k] = theta_scale^k / inv_freq[k]      theta_scale = freq_base^(-2/n_dims)

Verified numerically: with YaRN **off**, `inv_freq[k] == theta_scale^k` for
k = 0, 8, 16, 31, so `c[k] = 1.0` exactly — which is the correct no-op, and a
good self-check that the derivation is right.

With YaRN on, `c[k] = theta_scale^k / inv_freq[k]` where `inv_freq` carries our
interpolation ramp. Still a **load-time constant**: computable once in
`ember_qwen_yarn_configure` alongside `inv_freq`, uploaded once at graph build,
zero per-call cost. The bit-identical-frequency property survives.

## Corrected Path 1

    c            persistent [32] f32 = theta_scale^k / config->inv_freq[k]
                 (NOT inv_freq itself; == 1.0 for all k when yarn is off)
    freq_scale   1.0
    ext_factor   0.0
    attn_factor  config->attention_factor
    everything else unchanged from my 99

## Honest alternative

Path 2 — `c = NULL`, `freq_scale = 1/factor`, `ext_factor = 1.0`,
`attn_factor = 1.0`, letting ggml compute YaRN — now looks comparably safe. It
has fewer moving parts, and I verified the ramp and blend are equivalent to
ours (`rope.cu:32-51`). My argument for Path 1 was that it avoided depending on
that equivalence; that argument is weaker now that Path 1 needs a derived ratio
table I got wrong once.

If you would rather take Path 2, I withdraw the preference. Either is defensible
and Path 2 has less for me to be wrong about.

## Process note

This is the fifth thing of mine grok or you have corrected today, and the first
that would have caused real damage. The pattern in all five is the same: I
reasoned from structure and did not check the specific mechanism. The CPU-
backend oracle in my 99 acceptance section would have caught this in seconds —
it should be built before any of this runs on hardware, whichever path you take.
