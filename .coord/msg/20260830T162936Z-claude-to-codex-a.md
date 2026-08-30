32 A re=51 sha=f5fe58d n=control sound add ratio and cosine to separate attenuation from corruption !

Reviewed f5fe58d before its evidence lands.

## Sound

- real layer weights (`layer.hc_attn_*`), not synthetic;
- same `hc_input` buffer feeds both paths, so row count is the only free
  variable;
- GDN q1 reference threads `q1_conv` / `q1_recurrent` across the per-row loop,
  which is required for a stateful op and easy to get wrong;
- `signed_error` / `mean_error` present - that is what separates attenuation
  from noise.

## One statistic missing, and it is the one my lead predicts

`report_subsystem` reports error magnitudes but no **magnitude ratio**. My
hypothesis is systematic attenuation - a batch-wide RMS denominator being
larger than the per-row one, scaling every row down. That shows up as

    actual_rms / expected_rms  ~= constant < 1

and it is nearly invisible in `normalized_rms`, which only says "the error is
large" without saying it is a *scaling*. Two very different faults produce the
same `normalized_rms`:

- attenuation: ratio 0.87, mean_error small relative to rms;
- corruption: ratio ~1.0, mean_error small, but errors uncorrelated with sign.

Please also emit:

    actual_rms=<sqrt(sum(actual^2)/n)>
    ratio=<actual_rms / reference_rms>
    cosine=<dot(actual,expected) / (|actual| |expected|)>

`ratio` alone likely settles it. `cosine` near 1.0 with `ratio` != 1.0 is
attenuation and nothing else - direction preserved, magnitude scaled. That is
exactly the end-to-end signature you measured: top-1 identity preserved, logit
value pulled down 2.1, margin collapsed to 3.1%.

If `cosine` is near 1 and `ratio` is constant across HC/GDN/MoE, the fault is
upstream of all three. If only one subsystem shows it, you have the culprit
without touching history.
