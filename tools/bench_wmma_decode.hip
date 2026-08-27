// WMMA decode attention for gfx1151 (RDNA3.5), q=1 MLA: 64 query heads, one
// latent KV head, D=512. Measured against the scalar kernel and against the
// explicit graph path.
//
// Layout facts used here are verified, not assumed:
//   * Machine-readable source: AMD GPUOpen `machine-readable-isa/latest`,
//     downloaded 2026-08-27, archive SHA-256
//     82404f1126761b7877595b622afa7e1f311f2f41e89a3abe9aaf8ad045c082e2,
//     `amdgpu_isa_rdna3_5.xml` dated 2026-08-04. Its WMMA AB format requires
//     two replicated 16x16 matrices in wave32; the RDNA4 layout is not used.
//   * A frag: lane L holds ROW L%16, all 16 K values (ISA 7.9 VGPR-view figure;
//     AMD Matrix Instruction Calculator -a gfx1151; CK wmma_gemm.hpp).
//   * B frag: lane L holds COLUMN L%16.
//   * D frag: c_frag[i] = D[2i + lane/16][lane%16] -- lanes 0-15 even rows,
//     16-31 odd rows.
//   * A/B must be replicated across lanes 0-15 / 16-31 (ISA 7.9). Loading with
//     lane = tid%16 gives that for free.
//   * LDS: 32 banks x DWORD, bank = (addr/4)%32 (ISA 12.1).
//
// Both GEMMs are matrix-matrix even at q=1 because the 64 query heads share one
// KV head: operational intensity is H_qo/H_kv = 64 (FlashInfer).
//
//   hipcc -O3 --offload-arch=gfx1151 -o bench_wmma bench_wmma_decode.hip

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define D 512
#define NHEAD 64
#define NTHREADS 512
#define NWAVE (NTHREADS / 32)
#define TR 16                       // KV rows per tile == WMMA N
#define MT (NHEAD / 16)             // 4 WMMA M-tiles over heads
#define NT (D / 16)                 // 32 WMMA N-tiles over head_dim
#define NEG_INF (-3.402823466e38f)
// ISA 12.1: row stride of D halves is 256 DWORDs, a multiple of the 32 banks,
// so consecutive rows alias. +32 halves shifts each row by 16 banks.
#define KSTRIDE (D + 32)

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

#define HIP_CHECK(x)                                                          \
    do {                                                                      \
        hipError_t e = (x);                                                   \
        if (e != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,            \
                         hipGetErrorString(e));                               \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

__global__ __launch_bounds__(NTHREADS) void wmma_decode_partial(
        float        * __restrict__ part_acc,   // [split][NHEAD][D]
        float        * __restrict__ part_m,     // [split][NHEAD]
        float        * __restrict__ part_l,     // [split][NHEAD]
        const __half * __restrict__ q,          // [NHEAD][D]  (f16 for WMMA)
        const __half * __restrict__ k,          // [n_kv][D]   (V aliases K)
        const __half * __restrict__ mask,       // [n_kv]
        int n_kv, int n_split, float scale) {
    const int s   = blockIdx.x;
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int lane = tid % 32;
    const int l16 = lane % 16;      // fragment lane; 16-31 replicate 0-15
    const int half_id = lane / 16;  // 0 -> even rows of D, 1 -> odd rows

    const int rows_per_split = (n_kv + n_split - 1) / n_split;
    const int r_begin = s * rows_per_split;
    const int r_end = min(n_kv, r_begin + rows_per_split);

    extern __shared__ char smem[];
    float  * sc    = (float *) smem;                        // (NWAVE/MT) * NHEAD * TR
    __half * pt    = (__half *) (sc + (NWAVE / MT) * NHEAD * TR);
    float  * mrun  = (float *) (pt + NHEAD * TR);           // NHEAD
    float  * lrun  = mrun + NHEAD;
    float  * alph  = lrun + NHEAD;

    // O accumulator: 4 M-tiles x 32 N-tiles = 128 D-fragments over 8 waves
    // => 16 fragments per wave = 128 floats/lane. This is the floor for holding
    // a 64x512 output in registers (cf. FlashMLA's 32768-register note).
    float8 acc[MT][NT / NWAVE];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT / NWAVE; ++n) acc[m][n] = (float8)(0.0f);

    for (int i = tid; i < NHEAD; i += NTHREADS) { mrun[i] = NEG_INF; lrun[i] = 0.0f; }
    __syncthreads();

    for (int r0 = r_begin; r0 < r_end; r0 += TR) {
        const int rows = min(TR, r_end - r0);


        // ── QK: S[64 heads][TR rows] = Q * K^T, waves 0..3 own one M-tile ──
#ifndef DIAG_NO_QK
        {   // all 8 waves: wave w owns M-tile w%MT and k-half w/MT
            const int qk_m = wave % MT, qk_kh = wave / MT;   // NWAVE/MT k-chunks
            float8 s_acc = (float8)(0.0f);
            // Unrolled so the compiler can batch the global q loads and the LDS
            // K loads ahead of the WMMA chain instead of serialising
            // load->wmma->load->wmma at full memory latency.
            const int ksteps = (D / 16) / (NWAVE / MT);
#pragma unroll 8
            for (int kk = qk_kh * ksteps; kk < (qk_kh + 1) * ksteps; ++kk) {
                const __half * qrow = q + (size_t) (qk_m * 16 + l16) * D + kk * 16;
                // clamp instead of bounds-branch: rows past the end are zeroed
                // by the softmax (p == 0), so their K values never contribute
                const int krow_i = min(r0 + l16, n_kv - 1);
                const __half * krow = k + (size_t) krow_i * D + kk * 16;
                const half16 a_frag = *(const half16 *) qrow;  // A[l16][*]
                const half16 b_frag = *(const half16 *) krow;  // B[*][l16] = K^T
                // f32-output WMMA takes 3 operands; OPSEL exists only on the
                // 16-bit-output variants (ISA 7.9).
                s_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a_frag, b_frag, s_acc);
            }
            // D frag -> LDS as S[head][row]
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int head = qk_m * 16 + 2 * i + half_id;
                sc[qk_kh * NHEAD * TR + head * TR + l16] = s_acc[i];
            }
        }
#endif
        __syncthreads();

        // ── online softmax, one head per thread ──
#ifndef DIAG_NO_SOFTMAX
        {   // 4 lanes per head rather than 1 thread per head: the old form left
            // six of eight waves idle at the barrier and measured 33 us of 164.
            const int LPH = NTHREADS / NHEAD;   // lanes per head
            const int sh = tid / LPH, ss = tid % LPH;
            float tmax = NEG_INF;
            float v[TR * NHEAD / NTHREADS];
#pragma unroll
            for (int t = 0; t < TR * NHEAD / NTHREADS; ++t) {
                const int r = ss * (TR * NHEAD / NTHREADS) + t;
                float raw = 0.0f;
#pragma unroll
                for (int c = 0; c < NWAVE / MT; ++c)
                    raw += sc[c * NHEAD * TR + sh * TR + r];
                const float mv = (r < rows) ? __half2float(mask[r0 + r]) : -1.0e30f;
                v[t] = mv > -1.0e20f ? raw * scale + mv : NEG_INF;
                tmax = fmaxf(tmax, v[t]);
            }
#pragma unroll
            for (int off = 1; off < LPH; off <<= 1)
                tmax = fmaxf(tmax, __shfl_xor(tmax, off, LPH));
            const float m_old = mrun[sh];
            const float m_new = fmaxf(m_old, tmax);
            const float a = m_old == NEG_INF ? 0.0f : __expf(m_old - m_new);
            float sum = 0.0f;
#pragma unroll
            for (int t = 0; t < TR * NHEAD / NTHREADS; ++t) {
                const float pv = v[t] == NEG_INF ? 0.0f : __expf(v[t] - m_new);
                pt[sh * TR + ss * (TR * NHEAD / NTHREADS) + t] = (__half) pv;
                sum += pv;
            }
#pragma unroll
            for (int off = 1; off < LPH; off <<= 1) sum += __shfl_xor(sum, off, LPH);
            if (ss == 0) {
                mrun[sh] = m_new;
                lrun[sh] = lrun[sh] * a + sum;
                alph[sh] = a;
            }
        }
#endif
        __syncthreads();

#ifndef DIAG_NO_PV
        // ── rescale O, then PV: O[64][512] += P[64][TR] * V[TR][512] ──
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float al[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) al[i] = alph[m * 16 + 2 * i + half_id];
            const __half * prow = pt + (size_t) (m * 16 + l16) * TR;
            const half16 a_frag = *(const half16 *) prow;   // TR == 16 halves
#pragma unroll
            for (int n = 0; n < NT / NWAVE; ++n) {
                const int ntile = wave * (NT / NWAVE) + n;
#ifdef DIAG_CONTIG_BFRAG
                const half16 b_frag = *(const half16 *)
                    (k + (size_t) min(r0 + l16, n_kv - 1) * D + ntile * 16);
#else
                half16 b_frag;
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    b_frag[j] = (_Float16) k[(size_t) min(r0 + j, n_kv - 1) * D
                                             + ntile * 16 + l16];
#endif
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[m][n][i] *= al[i];
                acc[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a_frag, b_frag, acc[m][n]);
            }
        }
#endif
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int n = 0; n < NT / NWAVE; ++n) {
            const int ntile = wave * (NT / NWAVE) + n;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int head = m * 16 + 2 * i + half_id;
                part_acc[((size_t) s * NHEAD + head) * D + ntile * 16 + l16] =
                    acc[m][n][i];
            }
        }
    }
    if (tid < NHEAD) {
        part_m[(size_t) s * NHEAD + tid] = mrun[tid];
        part_l[(size_t) s * NHEAD + tid] = lrun[tid];
    }
}

__global__ __launch_bounds__(NTHREADS) void wmma_decode_combine(
        float * __restrict__ out, const float * __restrict__ pa,
        const float * __restrict__ pm, const float * __restrict__ pl,
        const float * __restrict__ sinks, int n_split) {
    const int h = blockIdx.x, tid = threadIdx.x;
    __shared__ float mm, ll;
    if (tid == 0) {
        float m = sinks ? sinks[h] : NEG_INF;
        for (int s = 0; s < n_split; ++s) m = fmaxf(m, pm[(size_t) s * NHEAD + h]);
        float l = (sinks && m > NEG_INF) ? __expf(sinks[h] - m) : 0.0f;
        for (int s = 0; s < n_split; ++s) {
            const float ms = pm[(size_t) s * NHEAD + h];
            if (ms == NEG_INF) continue;
            l += pl[(size_t) s * NHEAD + h] * __expf(ms - m);
        }
        mm = m; ll = l;
    }
    __syncthreads();
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    for (int d = tid; d < D; d += NTHREADS) {
        float v = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            const float ms = pm[(size_t) s * NHEAD + h];
            if (ms == NEG_INF) continue;
            v += pa[((size_t) s * NHEAD + h) * D + d] * __expf(ms - mm);
        }
        out[(size_t) h * D + d] = v * inv;
    }
}

// ── CPU reference (double precision), same semantics as the graph path ──
static void reference(std::vector<float> & out, const std::vector<float> & q,
                      const std::vector<__half> & k, const std::vector<__half> & mask,
                      const std::vector<float> & sinks, int n_kv, float scale) {
    out.assign((size_t) NHEAD * D, 0.0f);
    std::vector<double> p(n_kv);
    for (int h = 0; h < NHEAD; ++h) {
        double m = sinks.empty() ? -INFINITY : sinks[h];
        for (int r = 0; r < n_kv; ++r) {
            const float mv = __half2float(mask[r]);
            if (mv <= -1.0e20f) { p[r] = -INFINITY; continue; }
            double dot = 0.0;
            for (int d = 0; d < D; ++d)
                dot += (double) q[(size_t) h * D + d] *
                       (double) __half2float(k[(size_t) r * D + d]);
            p[r] = dot * scale + mv;
            if (p[r] > m) m = p[r];
        }
        double den = sinks.empty() ? 0.0 : std::exp(sinks[h] - m);
        for (int r = 0; r < n_kv; ++r) {
            p[r] = p[r] == -INFINITY ? 0.0 : std::exp(p[r] - m);
            den += p[r];
        }
        for (int r = 0; r < n_kv; ++r) {
            if (p[r] == 0.0) continue;
            const double w = p[r] / den;
            for (int d = 0; d < D; ++d)
                out[(size_t) h * D + d] +=
                    (float) (w * __half2float(k[(size_t) r * D + d]));
        }
    }
}

// Between two layers' attention, production streams MoE expert weights, which
// evicts the KV span from MALL/L2. Benchmarking attention back-to-back leaves it
// resident and flatters every attention kernel. This kernel touches a buffer
// large enough to evict it, standing in for that traffic.
__global__ void evict_cache(float * __restrict__ buf, size_t n, float v) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = buf[i] * 1.000001f + v;
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 100;
    const size_t evict_mb = argc > 2 ? (size_t) atoi(argv[2]) : 64;
    struct Ref { int n_kv; double us; };
    const Ref refs[] = {{128, 40.1}, {256, 48.3}, {416, 66.7}, {512, 75.3},
                        {768, 101.6}, {960, 114.9}, {1616, 0}, {3278, 0},
                        {8896, 1178.4}};
    const double scalar_us[] = {20.3, 29.2, 39.2, 43.5, 57.7, 69.3, 110.6, 201.4, 512.1};

    const float scale = 1.0f / std::sqrt((float) D);
    int max_kv = 0;
    for (const Ref & r : refs) max_kv = std::max(max_kv, r.n_kv);
    const int MAXSPLIT = 64;

    std::vector<float> qf((size_t) NHEAD * D), sinks(NHEAD);
    std::vector<__half> qh((size_t) NHEAD * D), k((size_t) max_kv * D), mask(max_kv);
    unsigned seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u;
                       return ((float)(seed >> 8) / (float)(1 << 24) - 0.5f) * 2.0f; };
    for (size_t i = 0; i < qf.size(); ++i) { qf[i] = rnd() * 0.5f; qh[i] = __float2half(qf[i]); }
    for (auto & v : k) v = __float2half(rnd() * 0.5f);
    for (auto & v : sinks) v = rnd();
    // q is demoted to f16 for WMMA; compare the reference against what the
    // kernel actually consumes so the error reported is the kernel's, not the
    // demotion's. The demotion cost is reported separately below.
    std::vector<float> qf16(qf.size());
    for (size_t i = 0; i < qf.size(); ++i) qf16[i] = __half2float(qh[i]);

    float *dq_unused, *dsink, *dpa, *dpm, *dpl, *dout;
    __half *dqh, *dk, *dmask;
    HIP_CHECK(hipMalloc(&dqh, qh.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dk, k.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dmask, mask.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dsink, sinks.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dpa, (size_t) MAXSPLIT * NHEAD * D * sizeof(float)));
    HIP_CHECK(hipMalloc(&dpm, (size_t) MAXSPLIT * NHEAD * sizeof(float)));
    HIP_CHECK(hipMalloc(&dpl, (size_t) MAXSPLIT * NHEAD * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, (size_t) NHEAD * D * sizeof(float)));
    HIP_CHECK(hipMemcpy(dqh, qh.data(), qh.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dk, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsink, sinks.data(), sinks.size() * sizeof(float), hipMemcpyHostToDevice));
    (void) dq_unused;

    const size_t shmem = (size_t) (NWAVE / MT) * NHEAD * TR * sizeof(float)
                       + (size_t) NHEAD * TR * sizeof(__half)
                       + 3 * NHEAD * sizeof(float);
    std::printf("LDS per block: %zu bytes (%.1f KB)\n", shmem, shmem / 1024.0);
    if (shmem > 64u * 1024u) { std::printf("OVER LDS BUDGET\n"); return 1; }

    // eviction buffer: MALL on this part is 32 MB, so 64 MB guarantees the KV
    // span is gone by the next attention call
    float * devict = nullptr;
    const size_t evict_n = evict_mb * 1024 * 1024 / sizeof(float);
    HIP_CHECK(hipMalloc(&devict, evict_n * sizeof(float)));
    HIP_CHECK(hipMemset(devict, 0, evict_n * sizeof(float)));
    std::printf("cache-eviction stream between iterations: %zu MB\n", evict_mb);
    std::printf("%-8s %-6s %10s %10s %10s %9s %9s %9s\n", "n_kv", "split",
                "wmma_us", "scalar_us", "expl_full", "vs_scal", "vs_expl", "maxerr");
    for (int ri = 0; ri < (int) (sizeof(refs) / sizeof(refs[0])); ++ri) {
        const int n_kv = refs[ri].n_kv;
        for (int r = 0; r < n_kv; ++r)
            mask[r] = __float2half((r >= 120 && r < 128) ? -1.0e30f : 0.0f);
        HIP_CHECK(hipMemcpy(dmask, mask.data(), (size_t) n_kv * sizeof(__half),
                            hipMemcpyHostToDevice));

        double best = 1e30; int bsplit = 1;
        for (int split = 1; split <= MAXSPLIT; split *= 2) {
            if ((n_kv + split - 1) / split < TR) break;
            hipEvent_t a, b; HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
            for (int i = 0; i < 3; ++i) {
                wmma_decode_partial<<<split, NTHREADS, shmem>>>(
                    dpa, dpm, dpl, dqh, dk, dmask, n_kv, split, scale);
                wmma_decode_combine<<<NHEAD, NTHREADS>>>(dout, dpa, dpm, dpl, dsink, split);
            }
            if (hipDeviceSynchronize() != hipSuccess) { continue; }
            // time the eviction stream alone so it can be subtracted
            HIP_CHECK(hipEventRecord(a));
            for (int i = 0; i < iters; ++i)
                evict_cache<<<(evict_n + 255) / 256, 256>>>(devict, evict_n, 1.0f);
            HIP_CHECK(hipEventRecord(b)); HIP_CHECK(hipEventSynchronize(b));
            float ms_ev = 0; HIP_CHECK(hipEventElapsedTime(&ms_ev, a, b));

            HIP_CHECK(hipEventRecord(a));
            for (int i = 0; i < iters; ++i) {
                evict_cache<<<(evict_n + 255) / 256, 256>>>(devict, evict_n, 1.0f);
                wmma_decode_partial<<<split, NTHREADS, shmem>>>(
                    dpa, dpm, dpl, dqh, dk, dmask, n_kv, split, scale);
                wmma_decode_combine<<<NHEAD, NTHREADS>>>(dout, dpa, dpm, dpl, dsink, split);
            }
            HIP_CHECK(hipEventRecord(b)); HIP_CHECK(hipEventSynchronize(b));
            float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
            ms -= ms_ev;   // charge only the attention work
            // isolate the partial pass
            hipEvent_t c, d2; HIP_CHECK(hipEventCreate(&c)); HIP_CHECK(hipEventCreate(&d2));
            HIP_CHECK(hipEventRecord(c));
            for (int i = 0; i < iters; ++i)
                wmma_decode_partial<<<split, NTHREADS, shmem>>>(
                    dpa, dpm, dpl, dqh, dk, dmask, n_kv, split, scale);
            HIP_CHECK(hipEventRecord(d2)); HIP_CHECK(hipEventSynchronize(d2));
            float msp = 0; HIP_CHECK(hipEventElapsedTime(&msp, c, d2));
            HIP_CHECK(hipEventDestroy(c)); HIP_CHECK(hipEventDestroy(d2));
            if (n_kv == 8896)
                std::printf("    [split %2d] total %7.1f us = partial %7.1f + combine %7.1f\n",
                            split, ms * 1000.0 / iters, msp * 1000.0 / iters,
                            (ms - msp) * 1000.0 / iters);
            HIP_CHECK(hipEventDestroy(a)); HIP_CHECK(hipEventDestroy(b));
            const double us = ms * 1000.0 / iters;
            if (us < best) { best = us; bsplit = split; }
        }

        wmma_decode_partial<<<bsplit, NTHREADS, shmem>>>(
            dpa, dpm, dpl, dqh, dk, dmask, n_kv, bsplit, scale);
        wmma_decode_combine<<<NHEAD, NTHREADS>>>(dout, dpa, dpm, dpl, dsink, bsplit);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> got((size_t) NHEAD * D);
        HIP_CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));

        double maxerr = 0.0;
        if (n_kv <= 1024) {
            std::vector<float> want;
            std::vector<__half> ks(k.begin(), k.begin() + (size_t) n_kv * D);
            std::vector<__half> ms2(mask.begin(), mask.begin() + n_kv);
            reference(want, qf16, ks, ms2, sinks, n_kv, scale);
            for (size_t i = 0; i < want.size(); ++i)
                maxerr = std::max(maxerr, (double) std::fabs(got[i] - want[i]));
            // separately: what does demoting q to f16 for WMMA cost, against a
            // reference that keeps q in f32?
            std::vector<float> want32;
            reference(want32, qf, ks, ms2, sinks, n_kv, scale);
            double demote = 0.0;
            for (size_t i = 0; i < want.size(); ++i)
                demote = std::max(demote, (double) std::fabs(want32[i] - want[i]));
            std::printf("%-8s q-demotion error (f32 q vs f16 q, reference only): %.3g\n",
                        "", demote);
        }
        const double full = refs[ri].us;   // already the full-path cost
        std::printf("%-8d %-6d %10.1f %10.1f %10s %8.2fx %8s %9.2g\n",
                    n_kv, bsplit, best, scalar_us[ri],
                    refs[ri].us > 0 ? "" : "-", scalar_us[ri] / best,
                    refs[ri].us > 0 ? "" : "-", maxerr);
        if (refs[ri].us > 0)
            std::printf("%-8s %-6s %10s %10s %10.1f %9s %7.2fx\n", "", "", "", "",
                        full, "", full / best);
    }
    // ── whole-token walk: all 43 layers at their real spans ──
    // DS4-Flash: layers 0-1 uncompressed, even>=2 ratio-4, odd>=2 ratio-128
    // (deepseek4_loader.cpp compute_compress_ratios). Compressed rows pad to a
    // multiple of 16. Eviction between layers stands in for MoE expert traffic.
    {
        const int NLAYER = 43;
        const int positions[] = {4096, 12600, 35000, 65536};
        std::printf("\n=== whole-token attention, %d layers, %zu MB evicted between layers ===\n",
                    NLAYER, evict_mb);
        std::printf("%-9s %10s %12s %14s\n", "ctx_pos", "n_attn(r4)", "wmma_ms/tok", "note");
        for (int pi = 0; pi < 4; ++pi) {
            const int P = positions[pi];
            double total_us = 0.0;
            int n_attn_r4 = 0;
            for (int il = 0; il < NLAYER; ++il) {
                const int ratio = (il < 2) ? 0 : ((il & 1) == 0 ? 4 : 128);
                const int ncomp = ratio ? ((P / ratio + 15) / 16) * 16 : 0;
                const int n_kv = 128 + ncomp;
                if (ratio == 4) n_attn_r4 = n_kv;
                if (n_kv > max_kv) { total_us = -1.0; break; }
                for (int r = 0; r < n_kv; ++r)
                    mask[r] = __float2half((r >= 120 && r < 128) ? -1.0e30f : 0.0f);
                HIP_CHECK(hipMemcpy(dmask, mask.data(), (size_t) n_kv * sizeof(__half),
                                    hipMemcpyHostToDevice));
                int split = 1;
                while (split < 32 && n_kv / (split * 2) >= 256) split *= 2;
                const int it = 20;
                hipEvent_t a, b; HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
                for (int w = 0; w < 3; ++w)
                    wmma_decode_partial<<<split, NTHREADS, shmem>>>(
                        dpa, dpm, dpl, dqh, dk, dmask, n_kv, split, scale);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipEventRecord(a));
                for (int i = 0; i < it; ++i)
                    evict_cache<<<(evict_n + 255) / 256, 256>>>(devict, evict_n, 1.0f);
                HIP_CHECK(hipEventRecord(b)); HIP_CHECK(hipEventSynchronize(b));
                float mev = 0; HIP_CHECK(hipEventElapsedTime(&mev, a, b));
                HIP_CHECK(hipEventRecord(a));
                for (int i = 0; i < it; ++i) {
                    evict_cache<<<(evict_n + 255) / 256, 256>>>(devict, evict_n, 1.0f);
                    wmma_decode_partial<<<split, NTHREADS, shmem>>>(
                        dpa, dpm, dpl, dqh, dk, dmask, n_kv, split, scale);
                    wmma_decode_combine<<<NHEAD, NTHREADS>>>(dout, dpa, dpm, dpl, dsink, split);
                }
                HIP_CHECK(hipEventRecord(b)); HIP_CHECK(hipEventSynchronize(b));
                float mall = 0; HIP_CHECK(hipEventElapsedTime(&mall, a, b));
                HIP_CHECK(hipEventDestroy(a)); HIP_CHECK(hipEventDestroy(b));
                total_us += (mall - mev) * 1000.0 / it;
            }
            if (total_us < 0) { std::printf("%-9d %10s %12s %14s\n", P, "-", "-", "exceeds bench KV"); continue; }
            const char * note = (P == 35000) ? "vs 18.37 explicit" : "";
            std::printf("%-9d %10d %12.2f %14s\n", P, n_attn_r4, total_us / 1000.0, note);
        }
    }
    return 0;
}
