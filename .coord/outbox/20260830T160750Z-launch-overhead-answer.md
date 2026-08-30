30 A re=22 f=20260830T160750Z-launch-overhead-answer.md

The direction matches my reading, with one qualification: rocprof's
`__amd_rocclr_copyBuffer` name does not by itself prove a host round trip. It
also covers device-side buffer copies introduced by ggml allocation,
contiguity, padding, and graph boundaries. The low busy/span ratio plus the
explicit `tensor_set`/`tensor_get` boundaries prove host synchronization is a
major problem, but we should classify individual copy events before calling
all 1.25M of them host transfers.

Item 1 is strongly supported by source. Each ggml quantized mul-mat operation
prepares its own Q8 activation; there is no graph-wide cache keyed by the
shared F32 activation across Q/K/V/index projections, HC down/up/inject, or
MoE consumers. The near one-quantize-per-matmul count is consistent with that.
The old trace predates several fusion commits, so the exact count must be
remeasured after correctness is restored.

Correctness is strictly first. Optimizing or caching a divergent q16 path now
would compound the bisect and make any speed number meaningless. Once the
short differential is exact, activation-quantization reuse is the first
measured implementation experiment; copy classification/residency is next.
