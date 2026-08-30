#include "reduce_rows.cuh"
#include "sumrows.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    // Cap gridDim and let each block walk a strided run of rows. One block per
    // row makes launch cost scale with nrows, which is the dominant term for
    // short rows (DS4's [4 x 8.4M] hyper-connection sum). The per-row reduction
    // is unchanged, so this is bit-identical to the uncapped launch.
    const int64_t max_blocks = (int64_t) nsm * 128;
    const dim3 block_nums(nrows > max_blocks ? max_blocks : nrows, 1, 1);
    // The reduction tree must depend only on row width.  Selecting blockDim
    // from nrows made q=1 and batched GDN L2 normalization reduce the same 128
    // values through 512- and 32-thread trees respectively on gfx1151.
    // Short rows: the block-per-row path would give each lane one element and
    // spend the butterfly adding zeros. Bit-exact, see reduce_rows.cuh.
    if (reduce_rows_short_f32_cuda</*norm=*/false>(x, dst, ncols, nrows, stream)) {
        return;
    }
    const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
    reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, nrows);
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    // Cap gridDim and let each block walk a strided run of rows. One block per
    // row makes launch cost scale with nrows, which is the dominant term for
    // short rows (DS4's [4 x 8.4M] hyper-connection sum). The per-row reduction
    // is unchanged, so this is bit-identical to the uncapped launch.
    const int64_t max_blocks = (int64_t) nsm * 128;
    const dim3 block_nums(nrows > max_blocks ? max_blocks : nrows, 1, 1);
    // Keep the arithmetic tree invariant when only the number of independent
    // rows changes.  Grid capping may vary with nrows; block width may not.
    if (reduce_rows_short_f32_cuda</*norm=*/false>(src0_d, dst_d, ncols, nrows, stream)) {
        return;
    }
    const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
    reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols, nrows);
}
