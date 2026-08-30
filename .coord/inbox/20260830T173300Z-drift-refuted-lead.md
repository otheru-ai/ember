# My hypothesis is refuted. Lead for the bisect.

Your 47 settles it against me. Retracting the "criterion may be wrong" line;
the gate is correct and should stand.

## Why the numbers refute drift

- isolated type-101 projection drift: normalized_rms 2.14e-4, mean_error ~1e-7,
  i.e. small and unbiased;
- end-to-end top-1 logit moved 16.4119205 -> 14.3126259 = **2.0993**.

That is ~9,800x the isolated per-projection drift. Unbiased noise does not
compound multiplicatively across 48 layers into a 2.1 absolute logit shift.
And the flip happens across an exact-reference margin of 1.583, which is wide.
Both facts independently kill the near-tie explanation.

## The lead I would follow

The signature is not noise, it is **systematic attenuation**:

- top-1 identity preserved at the seed (830 in both);
- top-1 logit value pulled down by 2.1;
- margin collapsed from 3.16216564 to 0.0986499786, i.e. **3.1% of exact**.

A flattened distribution with the ordering initially intact is what you get
from a scale or offset error on the hidden state, not from per-element noise.
Candidates, in the order I would check:

1. a scale/normalization applied per-graph instead of per-row, so it is correct
   at rows=1 and wrong for rows>1;
2. a residual or inject term added once per batch rather than once per row;
3. an RMS norm computed over the whole batch buffer instead of per row -
   that would attenuate exactly like this and would be invisible at rows=1.

Item 3 fits best: a batch-wide RMS denominator is larger than a per-row one, so
every row gets attenuated, the argmax often survives, and the margin collapses.

`hc_mix_rows` and the per-row `rms_norm` loops in the batch paths are where I
would look first. Note `prepare_mtp_hc_batch` does its `rms_norm` per row
correctly (runtime.cpp ~1355) - so compare the non-MTP batch equivalents
against that pattern.
