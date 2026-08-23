#include "common.cuh"

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
