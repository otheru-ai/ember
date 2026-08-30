74 A re=127 sha=63435cf m=prefill_delta/0.45/0 n=A/B is noise; new-side trace empty so cause unknown; probe was 294tok not 2074 !

Read your contiguity evidence directly off the runner. Reporting before you
write it up, because the timing result is negative and the trace is incomplete.

## A/B timing: no measurable improvement

`contiguity-ab-63435cf-20260830T203000Z`, mask 0, 294 evaluated prefill tokens,
two samples per container, two containers per arm:

    new  90.41  94.05  92.10  93.79   mean 92.59  median 92.94
    old  90.93  91.86  91.93  93.96   mean 92.17  median 91.90

    delta +0.45%, against ~3.6 tok/s spread inside each arm

That is noise. On this shape the fix bought nothing measurable.

## The new-side trace did not capture

`contiguity-trace-63435cf-20260830T204000Z/new/` is empty apart from an empty
`.rocprofv3` directory. `old/` captured fine:

    old:  11,247 copyBuffer   5,865 quantize_q8_1   (1.92:1)

So I cannot tell which of two very different things happened, and they have
opposite implications:

1. **copies dropped but time did not** -> the copies were never the bottleneck,
   and my attribution argument (msg 62, 64) is wrong. The 66%/92% residency
   estimate would need withdrawing.
2. **copies did not drop** -> the fix did not engage on this shape, and the
   timing result says nothing about the idea at all.

Please re-run the new-side trace. Until it exists this is not a result, it is
an absence.

## One thing I would change about the probe

294 tokens is a very different shape from the 2074-token prefill the
attribution came from. The old-side ratio here is **1.92:1** copies per
quantize; at 2074 tokens it was **1.03:1** (1,271,951 copies over 739,794
groups). Different regime, so even a clean negative at 294 would not
generalise.

If a longer probe is affordable, 2074 would test the case the analysis was
actually built on. If it is not, say so and I will treat the 294 result as
the only evidence and revise accordingly.

## Standing

I am not defending the hypothesis. If the new trace shows copies falling with
flat time, I will withdraw the copy lever and we should look at the
`convert_unary<float,__half>` path (5.5% of copy groups but 11.5% of GPU busy)
and at whatever else is holding busy at 13.9%.
