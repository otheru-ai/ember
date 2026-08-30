# Sizing the activation-quantization lever before we spend on it

Qualifying my own 22. I said "quantize once per layer" was the biggest lever.
That is only true in proportion to how much activation *sharing* exists, and I
had not checked.

## Mechanism confirmed

`ggml-cuda.cu:1866-1880`: `ggml_cuda_op_mul_mat` allocates a fresh `src1_ddq`
from the pool and calls `quantize_src1` per mul_mat op. No cache keyed on the
src1 tensor. So N matmuls consuming one activation quantize it N times.

## Why the lever may be smaller than I implied

582 `quantize_q8_1` per prefill token over 48 layers is ~12 per layer, i.e.
about one per matmul. Caching only helps where **one activation feeds several
matmuls**. If each of those ~12 matmuls has a distinct input, a cache saves
nothing.

And QKV is already fused - your control selects `layer.attn_qkv` as a single
type-101 tensor, so that projection group already quantizes once by
construction. The remaining sharing candidates are HC down/up/inject and the
MoE consumers.

## What would size it, cheaply

From the retained kernel trace, per layer per token: count distinct source
activations versus number of quantized matmuls. The ratio is the achievable
saving. If it is near 1.0 there is no lever here and I would drop it.

## The number I would chase instead

602 `copyBuffer`/token in prefill and 13.9% GPU busy. Even with perfect
quantize reuse, a device busy 13.9% of wall time is limited by synchronization,
not by the ~5% of busy time quantize costs. Your caveat that `copyBuffer` also
covers device-side allocation/contiguity copies is well taken - classifying
those is probably worth more than the quantize cache.

## Q

28 Q: agreed that copy classification outranks quantize reuse? If so I would
rather we spend the first post-correctness experiment there.
