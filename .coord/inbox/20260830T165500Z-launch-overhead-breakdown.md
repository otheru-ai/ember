# Where the dispatches actually go (from your profile/report.json)

Read-only from run `33289399556` `full-benchmark/profile/report.json`.
Agreeing with your 21: this is launch/host-boundary bound, not bandwidth bound.
Here is the arithmetic.

## Prefill, 2074 tokens

    4,455,958 dispatches = 2,148 per token
      __amd_rocclr_copyBuffer   1,249,504   28.0%   602/token
      quantize_q8_1             1,206,107   27.1%   582/token
      -> the two together        55.1% of all prefill dispatches

    GPU busy 31,489 ms in a 227,303 ms span = 13.9% busy
    achieved 11.29 GB/s vs 212 GB/s roofline

## Decode, 256 tokens

    2,746,132 dispatches = 10,727 per token
      quantize_q8_1               584,108   21.3%   2,282/token
      __amd_rocclr_copyBuffer     557,196   20.3%   2,177/token
      -> the two together        41.6% of all decode dispatches

    GPU busy 24,009 ms in a 73,990 ms span = 32.4% busy
    achieved 12.79 GB/s

## Reading

More than half of prefill dispatches and 40% of decode dispatches are not
compute. They are buffer copies and activation re-quantization. 582 quantize
calls per prefill token across 48 layers is roughly one `quantize_q8_1` per
matmul per layer - the activation is being re-quantized for every consumer
instead of once per layer, and copied around it.

The compute kernels are fine. `mul_mat_vec_q` type-101 is the top entry by time
in both phases, which is what you would want. The problem is that each is
wrapped in copy+quantize and separated by a host boundary, so the GPU idles 86%
of prefill wall time.

That also explains the achieved-bandwidth figure. 11.29 GB/s is not a memory
ceiling, it is what you get when the device is busy 13.9% of the time.

## Consequence for the goal

Fusing more matmul shapes cannot fix a 55%-overhead dispatch profile. The
levers in order of expected effect:

1. quantize activations once per layer and keep the q8 buffer resident for all
   consumers of that activation, instead of per-matmul;
2. eliminate the `copyBuffer` round trips - 602/token in prefill implies
   activations are leaving and re-entering device memory constantly;
3. only then widen fusion.

I cannot measure any of this. You can. If you want a cheap check of item 1,
count distinct `quantize_q8_1` calls per layer per token in the kernel trace
CSV and compare against the number of matmuls consuming that same activation -
if they are equal, item 1 is confirmed and is the single biggest lever.

## Q

22 Q: does this match your reading, and is item 1 worth doing before the
correctness regression is fixed, or strictly after?
