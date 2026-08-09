// Decode-shaped D=512 MLA attention: flash-decoding prototype + CPU reference.
//
// Why this exists: the production D=512 kernel reuses K/V across the *query*
// dimension (grid_x), so at q=1 it re-reads the whole KV span once per
// head-block — measured 115-148 GB/s of pure re-read, 2.9-5.4x slower per layer
// than the explicit GEMM pair it would replace. Its shared memory is
// O(HEADS_PER_BLOCK * n_kv), which is what forces the head group down to 4/2/1
// exactly when the span gets long.
//
// This prototype inverts that: shared memory is O(TILE_ROWS * D), independent of
// n_kv, so a block can hold G heads and stream the span in tiles with an online
// softmax. KV is then read n_heads/G times instead of n_heads times, and split-K
// supplies the parallelism that one block per head-group would otherwise lose.
//
// K is consumed as F16 (its storage type) while q stays F32 and accumulation is
// F32 throughout -- strictly better numerics than the F16-GEMM routes, which
// round q entering the WMMA tiles.
//
//   hipcc -O3 --offload-arch=gfx1151 -o bench_decode_attn bench_decode_attn.hip

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define D 512
#define NTHREADS 256
#define NEG_INF (-3.402823466e38f)

#define HIP_CHECK(x)                                                          \
    do {                                                                      \
        hipError_t err__ = (x);                                               \
        if (err__ != hipSuccess) {                                            \
            std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                         hipGetErrorString(err__));                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ─────────────────────────── partial pass ───────────────────────────
// grid(n_split, n_heads/G), block(256).
// Each block owns G heads and one contiguous KV slice, and emits an
// unnormalised accumulator plus the running (m, l) for that slice.
template <int G, int TR>
__global__ __launch_bounds__(NTHREADS) void ds4_decode_attn_partial(
        float        * __restrict__ part_acc,
        float        * __restrict__ part_m,
        float        * __restrict__ part_l,
        const float  * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ mask,
        int n_heads, int n_kv, int n_split, float scale) {
    constexpr int TPR = NTHREADS / TR;  // threads per KV row: no idle lanes
    constexpr int DPT = D / TPR;        // dims each thread folds into a dot

    const int s   = blockIdx.x;
    const int hg  = blockIdx.y;
    const int tid = threadIdx.x;
    const int h0  = hg * G;

    const int rows_per_split = (n_kv + n_split - 1) / n_split;
    const int r_begin        = s * rows_per_split;
    const int r_end          = min(n_kv, r_begin + rows_per_split);

    // LDS holds only the K tile and the score scratch: O(TR*D), never O(n_kv).
    // q stays in global (it is 2 KB per head and stays cached); staging it here
    // costs more in occupancy than it saves in reads.
    extern __shared__ char smem_raw[];
    __half * ktile = (__half *) smem_raw;              // TR * D
    float  * sc    = (float *) (ktile + (size_t) TR * D);
    float  * mrun  = sc + (size_t) G * TR;
    float  * lrun  = mrun + G;
    float  * alpha = lrun + G;

    float2 acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = make_float2(0.0f, 0.0f);
    if (tid < G) { mrun[tid] = NEG_INF; lrun[tid] = 0.0f; }
    __syncthreads();

    const int r_local = tid / TPR;
    const int lane    = tid % TPR;

    for (int r0 = r_begin; r0 < r_end; r0 += TR) {
        const int rows = min(TR, r_end - r0);

        const __half2 * ksrc = (const __half2 *) (k + (size_t) r0 * D);
        __half2 * kdst = (__half2 *) ktile;
        for (int i = tid; i < rows * (D / 2); i += NTHREADS) kdst[i] = ksrc[i];
        __syncthreads();

        for (int h = 0; h < G; ++h) {
            float partial = 0.0f;
            if (r_local < rows && h0 + h < n_heads) {
                // Interleaved lane->dim mapping, half2 loads. The blocked form
                // (d = lane*DPT + t) puts the TPR lanes 64 B apart, which is an
                // 8-way LDS bank conflict; striding by TPR puts them on
                // consecutive 4 B words instead. Same dims, same F32 accumulate.
                const float2  * q2 =
                    (const float2 *) (q + (size_t) (h0 + h) * D);
                const __half2 * k2 =
                    (const __half2 *) (ktile + (size_t) r_local * D);
#pragma unroll
                for (int t = 0; t < DPT / 2; ++t) {
                    const int idx = t * TPR + lane;
                    const float2 qq = q2[idx];
                    const float2 kk = __half22float2(k2[idx]);
                    partial += qq.x * kk.x + qq.y * kk.y;
                }
            }
#pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1) {
                partial += __shfl_down(partial, off, TPR);
            }
            if (lane == 0 && r_local < rows) {
                const float mv = __half2float(mask[r0 + r_local]);
                sc[h * TR + r_local] =
                    mv > -1.0e20f ? partial * scale + mv : NEG_INF;
            }
        }
        __syncthreads();

        if (tid < G && h0 + tid < n_heads) {
            float tile_max = NEG_INF;
            for (int r = 0; r < rows; ++r) {
                tile_max = fmaxf(tile_max, sc[tid * TR + r]);
            }
            const float m_old = mrun[tid];
            const float m_new = fmaxf(m_old, tile_max);
            const float a = m_old == NEG_INF ? 0.0f : __expf(m_old - m_new);
            float sum = 0.0f;
            for (int r = 0; r < rows; ++r) {
                const float p = sc[tid * TR + r] == NEG_INF
                    ? 0.0f : __expf(sc[tid * TR + r] - m_new);
                sc[tid * TR + r] = p;
                sum += p;
            }
            mrun[tid]  = m_new;
            lrun[tid]  = lrun[tid] * a + sum;
            alpha[tid] = a;
        }
        __syncthreads();

        const __half2 * kpair = (const __half2 *) ktile;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float a = alpha[h];
            acc[h].x *= a;
            acc[h].y *= a;
            for (int r = 0; r < rows; ++r) {
                const float p = sc[h * TR + r];
                if (p == 0.0f) continue;
                const float2 kk =
                    __half22float2(kpair[(size_t) r * (D / 2) + tid]);
                acc[h].x += p * kk.x;
                acc[h].y += p * kk.y;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int h = 0; h < G; ++h) {
        const int hh = h0 + h;
        if (hh >= n_heads) continue;
        float * dst = part_acc + ((size_t) s * n_heads + hh) * D;
        dst[2 * tid + 0] = acc[h].x;
        dst[2 * tid + 1] = acc[h].y;
        if (tid == 0) {
            part_m[(size_t) s * n_heads + hh] = mrun[h];
            part_l[(size_t) s * n_heads + hh] = lrun[h];
        }
    }
}

// ─────────────────────────── combine pass ───────────────────────────
// Merges the split partials and applies the sink, which contributes to the
// denominator only -- so it must be added exactly once, here.
__global__ __launch_bounds__(NTHREADS) void ds4_decode_attn_combine(
        float       * __restrict__ out,       // [n_heads][D]
        const float * __restrict__ part_acc,
        const float * __restrict__ part_m,
        const float * __restrict__ part_l,
        const float * __restrict__ sinks,
        int n_heads, int n_split) {
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float m_all, l_all;
    if (tid == 0) {
        float m = sinks ? sinks[h] : NEG_INF;
        for (int s = 0; s < n_split; ++s) {
            m = fmaxf(m, part_m[(size_t) s * n_heads + h]);
        }
        float l = sinks && m > NEG_INF ? __expf(sinks[h] - m) : 0.0f;
        for (int s = 0; s < n_split; ++s) {
            const float ms = part_m[(size_t) s * n_heads + h];
            if (ms == NEG_INF) continue;
            l += part_l[(size_t) s * n_heads + h] * __expf(ms - m);
        }
        m_all = m;
        l_all = l;
    }
    __syncthreads();

    const float inv = l_all > 0.0f ? 1.0f / l_all : 0.0f;
    for (int d = tid; d < D; d += NTHREADS) {
        float v = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            const float ms = part_m[(size_t) s * n_heads + h];
            if (ms == NEG_INF) continue;
            v += part_acc[((size_t) s * n_heads + h) * D + d] *
                 __expf(ms - m_all);
        }
        out[(size_t) h * D + d] = v * inv;
    }
}

// ─────────────────────────── CPU reference ───────────────────────────
// Mirrors the explicit graph path: scaled scores + additive mask, sink logit in
// the denominator only, dense over every visible row.
static void reference(std::vector<float> & out, const std::vector<float> & q,
                      const std::vector<__half> & k,
                      const std::vector<__half> & mask,
                      const std::vector<float> & sinks,
                      int n_heads, int n_kv, float scale) {
    out.assign((size_t) n_heads * D, 0.0f);
    std::vector<double> p(n_kv);
    for (int h = 0; h < n_heads; ++h) {
        double m = sinks.empty() ? -INFINITY : sinks[h];
        for (int r = 0; r < n_kv; ++r) {
            const float mv = __half2float(mask[r]);
            if (mv <= -1.0e20f) { p[r] = -INFINITY; continue; }
            double dot = 0.0;
            for (int d = 0; d < D; ++d) {
                dot += (double) q[(size_t) h * D + d] *
                       (double) __half2float(k[(size_t) r * D + d]);
            }
            p[r] = dot * scale + mv;
            if (p[r] > m) m = p[r];
        }
        double denom = sinks.empty() ? 0.0 : std::exp(sinks[h] - m);
        for (int r = 0; r < n_kv; ++r) {
            p[r] = p[r] == -INFINITY ? 0.0 : std::exp(p[r] - m);
            denom += p[r];
        }
        for (int r = 0; r < n_kv; ++r) {
            if (p[r] == 0.0) continue;
            const double w = p[r] / denom;
            for (int d = 0; d < D; ++d) {
                out[(size_t) h * D + d] +=
                    (float) (w * __half2float(k[(size_t) r * D + d]));
            }
        }
    }
}

template <int G, int TR>
static double run(int n_heads, int n_kv, int n_split, const float * dq,
                  const __half * dk, const __half * dmask, const float * dsink,
                  float * dpacc, float * dpm, float * dpl, float * dout,
                  float scale, int iters) {
    const size_t shmem =
        (size_t) TR * D * sizeof(__half) + ((size_t) G * TR + 3 * G) * sizeof(float);
    // A config that overruns LDS silently fails to launch and then looks
    // infinitely fast. Reject it up front, and verify the launch below.
    if (shmem > 64u * 1024u) return 1e30;
    dim3 grid(n_split, (n_heads + G - 1) / G);
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) {
        ds4_decode_attn_partial<G, TR><<<grid, NTHREADS, shmem>>>(
            dpacc, dpm, dpl, dq, dk, dmask, n_heads, n_kv, n_split, scale);
        if (hipGetLastError() != hipSuccess) return 1e30;
        ds4_decode_attn_combine<<<n_heads, NTHREADS>>>(
            dout, dpacc, dpm, dpl, dsink, n_heads, n_split);
        if (hipGetLastError() != hipSuccess) return 1e30;
    }
    if (hipDeviceSynchronize() != hipSuccess) return 1e30;
    HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) {
        ds4_decode_attn_partial<G, TR><<<grid, NTHREADS, shmem>>>(
            dpacc, dpm, dpl, dq, dk, dmask, n_heads, n_kv, n_split, scale);
        ds4_decode_attn_combine<<<n_heads, NTHREADS>>>(
            dout, dpacc, dpm, dpl, dsink, n_heads, n_split);
    }
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return ms * 1000.0 / iters;   // microseconds
}

int main(int argc, char ** argv) {
    const int n_heads = 64;
    const int n_swa   = 128;
    const int iters   = argc > 1 ? atoi(argv[1]) : 200;

    // measured explicit-path cost per layer (score+value GEMM), from the
    // rocprofv3 baseline trace; 8896 from the earlier production decode trace.
    struct Ref { int n_kv; double us; };
    const Ref refs[] = {{128, 25.5}, {160, 26.5}, {256, 30.7}, {416, 42.4},
                        {512, 47.9}, {768, 64.6}, {960, 73.1}, {1616, 0.0},
                        {3278, 0.0}, {8896, 749.6}};

    const float scale = 1.0f / std::sqrt((float) D);
    int max_kv = 0;
    for (const Ref & r : refs) max_kv = std::max(max_kv, r.n_kv);
    const int max_split = 64;

    std::vector<float>  q((size_t) n_heads * D);
    std::vector<__half> k((size_t) max_kv * D);
    std::vector<__half> mask((size_t) max_kv);
    std::vector<float>  sinks(n_heads);
    unsigned seed = 12345;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float) (seed >> 8) / (float) (1 << 24) - 0.5f) * 2.0f;
    };
    for (auto & v : q) v = rnd() * 0.5f;
    for (auto & v : k) v = __float2half(rnd() * 0.5f);
    for (auto & v : sinks) v = rnd();

    float *dq, *dsink, *dpacc, *dpm, *dpl, *dout;
    __half *dk, *dmask;
    HIP_CHECK(hipMalloc(&dq, q.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dk, k.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dmask, mask.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dsink, sinks.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dpacc, (size_t) max_split * n_heads * D * sizeof(float)));
    HIP_CHECK(hipMalloc(&dpm, (size_t) max_split * n_heads * sizeof(float)));
    HIP_CHECK(hipMalloc(&dpl, (size_t) max_split * n_heads * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, (size_t) n_heads * D * sizeof(float)));
    HIP_CHECK(hipMemcpy(dq, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dk, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsink, sinks.data(), sinks.size() * sizeof(float), hipMemcpyHostToDevice));

    std::printf("%-8s %-9s %10s %10s %9s %10s %8s\n", "n_kv", "G/TR/spl",
                "flash_us", "expl_full", "speedup", "GB/s", "maxerr");
    for (const Ref & ref : refs) {
        const int n_kv = ref.n_kv;
        // full-ring path: rows [n_valid, n_swa) masked, compressed span visible
        for (int r = 0; r < n_kv; ++r) {
            mask[r] = __float2half((r >= n_swa - 8 && r < n_swa) ? -1.0e30f : 0.0f);
        }
        HIP_CHECK(hipMemcpy(dmask, mask.data(), (size_t) n_kv * sizeof(__half),
                            hipMemcpyHostToDevice));

        double best = 1e30;
        int best_split = 1, best_g = 16, best_tr = 16;
        const bool verbose = getenv("VERBOSE") && n_kv == 8896;
        auto sweep = [&](auto tag, int G, int TR) {
            double cbest = 1e30; int cspl = 0;
            for (int split = 1; split <= max_split; split *= 2) {
                if ((n_kv + split - 1) / split < TR) break;
                const double us = tag(n_heads, n_kv, split, dq, dk, dmask, dsink,
                                      dpacc, dpm, dpl, dout, scale, iters);
                if (us < cbest) { cbest = us; cspl = split; }
                if (us < best) { best = us; best_split = split; best_g = G; best_tr = TR; }
            }
            if (verbose && cbest < 1e29) {
                const double bytes = (double) (n_heads / G) * n_kv * D * 2.0;
                std::printf("    cfg G=%-3d TR=%-3d split=%-3d  %8.1f us  traffic=%6.1f MB  %5.0f GB/s\n",
                            G, TR, cspl, cbest, bytes / 1e6, bytes / cbest / 1e3);
            }
        };
        sweep(run<8, 16>,  8, 16);
        sweep(run<16, 16>, 16, 16);
        sweep(run<32, 16>, 32, 16);
        sweep(run<64, 16>, 64, 16);
        sweep(run<16, 8>,  16, 8);
        sweep(run<32, 8>,  32, 8);
        sweep(run<64, 8>,  64, 8);
        sweep(run<8, 32>,   8, 32);
        if (best_g == 64 && best_tr == 16) run<64, 16>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        else if (best_g == 64 && best_tr == 8) run<64, 8>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        else if (best_g == 8 && best_tr == 32) run<8, 32>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        else if (best_g == 8  && best_tr == 16) run<8, 16>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        else if (best_g == 32 && best_tr == 16) run<32, 16>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        else if (best_g == 16 && best_tr == 8)  run<16, 8>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        else if (best_g == 32 && best_tr == 8)  run<32, 8>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        else if (best_g == 16 && best_tr == 32) run<16, 32>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        else run<16, 16>(n_heads, n_kv, best_split, dq, dk, dmask, dsink, dpacc, dpm, dpl, dout, scale, 1);
        std::vector<float> got((size_t) n_heads * D);
        HIP_CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        double maxerr = 0.0;
        if (n_kv <= 1024) {
            std::vector<float> want;
            std::vector<__half> ksub(k.begin(), k.begin() + (size_t) n_kv * D);
            std::vector<__half> msub(mask.begin(), mask.begin() + n_kv);
            reference(want, q, ksub, msub, sinks, n_heads, n_kv, scale);
            for (size_t i = 0; i < want.size(); ++i) {
                maxerr = std::max(maxerr, (double) std::fabs(got[i] - want[i]));
            }
        }
        const double gb = (double) (n_heads / best_g) * n_kv * D * 2.0 / best / 1e3;
        const double full = ref.us * 1.572;   // measured non-GEMM share of the path
        if (ref.us > 0.0) {
            char cfg[32]; std::snprintf(cfg, sizeof(cfg), "%d/%d/%d", best_g, best_tr, best_split);
            std::printf("%-8d %-9s %10.1f %10.1f %8.2fx %10.0f %8.2g\n", n_kv,
                        cfg, best, full, full / best, gb, maxerr);
        } else {
            char cfg2[32]; std::snprintf(cfg2, sizeof(cfg2), "%d/%d/%d", best_g, best_tr, best_split);
            std::printf("%-8d %-9s %10.1f %10s %9s %10.0f %8.2g\n", n_kv,
                        cfg2, best, "-", "-", gb, maxerr);
        }
    }
    return 0;
}
