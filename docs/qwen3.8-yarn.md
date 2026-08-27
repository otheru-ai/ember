# Qwen3.8-Flash-Next static YaRN

Ember keeps Qwen3.8-Flash-Next at its native 262,144-token context unless an
operator explicitly passes both:

```text
--qwen-yarn --max-ctx 1000000
```

This resolves the official static recipe exactly: factor 4, original context
262,144, RoPE theta 10,000,000, partial rotary factor 0.25 (64 dimensions of a
256-wide head), and interleaved M-RoPE sections `[11, 11, 10]`. It is sourced
from the immutable
[Qwen model README](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/README.md#processing-ultra-long-texts).
The checkpoint's own metadata still says ordinary RoPE, so loading the model
card or choosing a context above 262,144 never turns YaRN on implicitly.

The initial policy intentionally accepts only the published 1,000,000-token
target. It does not silently apply factor 4 to an arbitrary extension; Qwen's
README specifically recommends factor 2 for a typical 524,288-token context.

## Memory gate

Correct positional arithmetic does not establish that a context is resident
or performant on Strix Halo. Qwen's q=1 QSA/index caches consume 55,296 bytes
per token, or 55,296,000,000 bytes at one million tokens, before recurrent
state and Ember's 8-GiB runtime reserve. Combined with roughly 94 GB of 4-bit
weights, the current plan exceeds 128 GiB. The loader therefore reports the
component budget and refuses the allocation before model residency or cache
growth. A future lower-residency weight format or cache implementation may
make the same correctness path usable, but it needs separate device validation
and performance measurements before any 1M capability claim.

Static YaRN also changes frequency and attention scaling at short positions.
Enable it only for a workload that needs the extended context; restart with the
native default for ordinary requests.
