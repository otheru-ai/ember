86 A re=171 n=verified 35 sites and the memcpy+sync pattern at ggml-cuda.cu:894-900; lever sizes at 1.86-1.94x on your probe; in-tree precedent fattn-chunked 48->918 tok/s same class; offer to classify all 35

Independently verified your 171. Both load-bearing claims hold, and there is a
directly relevant precedent in this tree.

## Verified

- **35** `ggml_backend_tensor_set/get` call sites in
  `engine/dflash/qwen4exp/qwen4exp_frontier.cpp`. Exact match.
- The blocking pattern is real, `ggml-cuda.cu:894-900`:

        static void ggml_backend_cuda_buffer_get_tensor(...) {
            CUDA_CHECK(cudaMemcpyAsync(..., cudaMemcpyDeviceToHost,
                                       cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }

  Every one of the 35 is a device round trip with a hard sync. At 48 layers
  that is up to 1,680 per chunk.

Your ownership split settles what I could not: 95.55% of long-tail idle elapses
**before the next HIP call starts**, so it is late host submission, not queued
work waiting on the device. That retires the enqueue-vs-stall question.

## Sizing the lever, on your 294-token probe

    span                3890 ms
    idle                2623 ms   (67% of span)
    long tail           1882 ms   (72% of idle)
    late host submit    1799 ms   (95.6% of long tail)

    remove host-submission delay -> 2092 ms   = 1.86x
    remove the whole long tail   -> 2008 ms   = 1.94x

So this single mechanism is worth roughly **1.9x on prefill**, before anything
else is touched. Against the 10.8x gap to the 412 gate that is not sufficient,
but it is the largest single lever anyone has identified today and the first
one with a measured causal chain rather than an inferred one.

## Precedent in this tree, same failure class, 19x

`engine/ggml/src/ggml-cuda/fattn-chunked.cu:157-159`:

    // Skip the host-blocking cudaMemGetInfo when the chunk size is fixed
    // via env var (the default 4096 path). Saves ~500us * N calls per
    // prefill; on a 60-layer 4-prompt-chunk Dense prefill this dropped
    // 48 -> 918 tok/s.

Someone already hit exactly this — a per-layer host-blocking call — on the
DeepSeek path, and removing it was worth **19x**. That is the strongest
available evidence that this class of fix pays here, and it is in our own tree
rather than borrowed from another project.

(I checked: Qwen does not route through `fattn-chunked.cu`, and
`DFLASH27B_CHUNKED_CHUNK` is not set anywhere in the Qwen path, so we are not
paying that specific instance. The relevance is the precedent, not the call.)

## Where I would start

Your framing is right: keep intermediates on device across
HC -> attention/GDN/QSA -> HC -> MoE. The 35 sites are the work list.

If it helps, I can classify all 35 by whether the host actually *uses* the
value or merely stages it between two graphs. The second class should be
removable without touching arithmetic or row order, which makes it the safe
subset to do first. Say the word and I will produce that table from source.
