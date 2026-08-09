# Vulkan coopmat vs hand-written WMMA — MLA decode attention on gfx1151

Head-to-head for the same kernel: q=1 MLA decode, 64 query heads sharing one
latent KV head, D=512, split-K, online softmax, f16 in / f32 accumulate.

- `mla_decode.comp` — VK_KHR_cooperative_matrix implementation (RADV lowers
  `coopMatMulAdd` to the same `v_wmma_f32_16x16x16_f16`).
- `vk_bench.cpp` — host harness; times with timestamp queries and validates
  against a double-precision reference over the exact f16 inputs.
- `vk_coopmat_probe.cpp` — enumerates coopmat configurations.

The HIP side is `../bench_wmma_decode.hip`.

## Build and run (no host packages needed)

    docker build -t vk-dev:f44 - <<'D'
    FROM fedora:44
    RUN dnf install -y --setopt=install_weak_deps=False vulkan-headers vulkan-loader \
        vulkan-loader-devel vulkan-tools mesa-vulkan-drivers glslc glslang gcc-c++ make \
     && dnf clean all
    WORKDIR /w
    D
    docker run --rm --device=/dev/dri --group-add video --group-add render \
      -v $PWD:/w -w /w vk-dev:f44 bash -lc \
      "glslc --target-env=vulkan1.3 -O -o mla.spv mla_decode.comp && \
       g++ -O2 -o vk_bench vk_bench.cpp -lvulkan && ./vk_bench 50"

## Device facts (RADV STRIX_HALO)

    subgroupSize                = 64        (not 32 -- RADV runs wave64 for compute)
    maxComputeSharedMemorySize  = 65536 B
    coopmat configurations      = 14, all M16 N16 K16, scope=subgroup
                                  incl. f16 x f16 -> f32

## Result

Both implementations validated to the same tolerance (maxerr 1.8e-05 .. 6e-06).

| n_kv | Vulkan coopmat | HIP WMMA | ratio |
|-----:|---------------:|---------:|------:|
|  128 |      15.0 us   |  14.6 us | 1.03x |
|  256 |      16.2 us   |  17.9 us | 0.90x |
|  416 |      26.9 us   |  21.6 us | 1.24x |
|  512 |      18.9 us   |  21.6 us | 0.88x |
|  768 |      29.6 us   |  24.9 us | 1.19x |
|  960 |      29.6 us   |  28.3 us | 1.05x |
| 1616 |      50.8 us   |  38.1 us | 1.33x |
| 3278 |      82.5 us   |  58.1 us | 1.42x |
| 8896 |     200.2 us   | 124.6 us | 1.61x |

HIP wins at every span from 416 up, by a margin that grows with context --
1.61x at 8896, which is the regime that dominates decode cost. Vulkan wins at
256 and 512.

One structural reason for the gap: coopmat1 exposes no element-to-(row,col)
mapping, so the per-head rescale of the online-softmax accumulator must
round-trip through shared memory. The HIP kernel knows `c_frag[i]` maps to row
`2i + lane/16` (ISA 7.9) and rescales in registers.
