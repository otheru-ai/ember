#include "common.cuh"

// ── Short rows: one thread per row, same arithmetic ─────────────────────────
//
// With blockDim 32 and ncols <= 8 the block-per-row kernel gives every lane at
// most one element, zero-fills the other 24+, and then runs the full five-step
// warp butterfly (strides 16,8,4,2,1) to add them. Almost all of that work is
// adding zeros, and it costs a whole wave per row.
//
// The same value can be produced by one thread in a handful of scalar adds, and
// BIT-EXACTLY so, because of two facts that hold only under strict IEEE (the
// HIP build keeps -ffast-math and -funsafe-math-optimizations off, see
// ggml-hip/CMakeLists.txt):
//
//   1. The block kernel forms each lane's value as `0.0f + x`, which maps -0.0
//      to +0.0. Reproducing that leaves no operand that can be negative zero.
//   2. For x that is not -0.0, `x + 0.0f` is exactly x. So every butterfly
//      stride at or above the padded width is an identity and folds away.
//
// What survives is a butterfly over P = next power of two >= ncols, in the same
// decreasing-stride order, which is what this computes. Worked example for
// ncols = 4, where lanes 4..31 are zero:
//   stride 16, 8, 4 -> a0 + 0        (identity)
//   stride 2        -> a0 + a2   and   a1 + a3
//   stride 1        -> (a0+a2) + (a1+a3)
// Three adds, matching the shuffle path term for term and in the same order.
//
// Only the halves of a stride below `s` are updated: the butterfly writes the
// same sum to lane i and lane i^s (IEEE addition is commutative), and lane 0
// never reads the upper halves again.
constexpr int reduce_rows_pow2_ceil(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

template <bool norm, int ncols_t>
static __global__ void reduce_rows_short_f32(const float * __restrict__ x, float * __restrict__ dst,
                                             const int nrows) {
    constexpr int P = reduce_rows_pow2_ceil(ncols_t);
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }

    float a[P];
#pragma unroll
    for (int i = 0; i < P; ++i) {
        // `0.0f +` is load-bearing: it reproduces the block kernel's
        // normalisation of -0.0 and must not be simplified away.
        a[i] = i < ncols_t ? 0.0f + x[(size_t) row * ncols_t + i] : 0.0f;
    }
#pragma unroll
    for (int s = P / 2; s >= 1; s >>= 1) {
#pragma unroll
        for (int i = 0; i < s; ++i) {
            a[i] += a[i + s];
        }
    }

    dst[row] = norm ? a[0] / ncols_t : a[0];
}

// Returns false when the shape is not one this path reproduces exactly.
template <bool norm>
static inline bool reduce_rows_short_f32_cuda(const float * x, float * dst, const int ncols,
                                              const int nrows, cudaStream_t stream) {
    if (ncols < 1 || ncols > 8 || nrows <= 0) {
        return false;
    }
    // Escape hatch for differential testing: the block-per-row path is still
    // compiled, so one binary can produce both results and they can be diffed.
    static const bool enabled = []() {
        const char * e = getenv("GGML_REDUCE_ROWS_SHORT");
        return !(e && e[0] == '0');
    }();
    if (!enabled) {
        return false;
    }
    constexpr int block = 256;
    const int64_t blocks = ((int64_t) nrows + block - 1) / block;
    dim3 nb((unsigned) blocks, 1, 1);
    switch (ncols) {
        case 1: reduce_rows_short_f32<norm, 1><<<nb, block, 0, stream>>>(x, dst, nrows); break;
        case 2: reduce_rows_short_f32<norm, 2><<<nb, block, 0, stream>>>(x, dst, nrows); break;
        case 3: reduce_rows_short_f32<norm, 3><<<nb, block, 0, stream>>>(x, dst, nrows); break;
        case 4: reduce_rows_short_f32<norm, 4><<<nb, block, 0, stream>>>(x, dst, nrows); break;
        case 5: reduce_rows_short_f32<norm, 5><<<nb, block, 0, stream>>>(x, dst, nrows); break;
        case 6: reduce_rows_short_f32<norm, 6><<<nb, block, 0, stream>>>(x, dst, nrows); break;
        case 7: reduce_rows_short_f32<norm, 7><<<nb, block, 0, stream>>>(x, dst, nrows); break;
        case 8: reduce_rows_short_f32<norm, 8><<<nb, block, 0, stream>>>(x, dst, nrows); break;
        default: return false;
    }
    return true;
}

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
// One block per row costs a block launch per row, which dominates when rows are
// short and numerous. DS4 prefill sums its 4 hyper-connection streams as a
// [4 x 8388608] sum_rows and spent 5.77 ms per call launching 8.4M blocks of 32
// threads to add four floats each -- 22 GB/s against a shape that should stream.
//
// `nrows` lets the caller cap gridDim and hand each block a strided run of rows.
// The per-row work is untouched: every row still reduces across the same lanes
// through the same block_reduce butterfly, so results stay bit-identical to the
// one-block-per-row launch. Passing nrows <= 0 keeps the original behaviour for
// callers that have not been updated.
template <bool norm>
static __global__ void reduce_rows_f32(const float * __restrict__ x, float * __restrict__ dst,
                                       const int ncols, const int nrows = 0) {
    const int col = threadIdx.x;
    const int row_end = nrows > 0 ? nrows : (int) (blockIdx.x + 1);

    for (int row = blockIdx.x; row < row_end; row += gridDim.x) {
    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                temp[j] = x[row * ncols + i];
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    // sum up partial sums
    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    if (col == 0) {
        dst[row] = norm ? sum / ncols : sum;
    }
    }
}
