#include "quantize.cuh"
#include "../../rocmfpx/w4a4_grid.h"
#include <cstdint>

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

template <mmq_q8_1_ds_layout ds_layout, bool i4_grid = false>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2,
        const int64_t group_width, const int64_t group_stride) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them.
    // A grouped source is physically [group_width, token, group] while this
    // kernel emits the same flattened-K Q8 layout as a materialized permute.
    int64_t src_i00 = i00;
    int64_t src_group_offset = 0;
    if (group_width > 0) {
        const int64_t group = i00 / group_width;
        src_i00 = i00 - group * group_width;
        src_group_offset = group * group_stride;
    }
    const int64_t src_index =
        i03*s03 + i02*s02 + i01*s01 + src_group_offset + src_i00;
    const float4 xi = i0 < ne00
        ? x4[src_index/4]
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    // Cached q5/q16 graphs zero-fill unused token rows.  Keep those rows a
    // real zero Q8 block: 0*inf is NaN and integer conversion of that NaN is
    // undefined even though the corresponding padded output is discarded.
    const float d_inv = amax > 0.0f
        ? (i4_grid ? 7.0f : 127.0f) / amax
        : 0.0f;
    if constexpr (i4_grid) {
        const uint32_t lo4 = rocmi4_w4a4_pack4(xi.x, xi.y, xi.z, xi.w, d_inv);
        const int hi4 = __shfl_xor_sync(0xffffffff, lo4, 1, WARP_SIZE);
        if (iqs % 8 == 0) {
            ((int *) y[ib].qs)[iqs/8] = (int)rocmi4_w4a4_fold(lo4, (uint32_t)hi4);
        }
    } else {
    char4 q;
    // EMBER FORK DIVERGENCE (engine/VENDOR.md), from llama.cpp issue #21284
    // "Inefficient defaults for gfx1151".  roundf() lowers to a call-like
    // sequence; __float2int_rn is a single v_cvt_i32_f32, round-to-nearest
    // being the default mode.  quantize_mmq_q8_1 measured 3.3% of GPU time over
    // 7,590 dispatches in a rocprofv3 trace, so these four calls are hot.
    //
    // NOT bit-identical: roundf() rounds halves away from zero, __float2int_rn
    // rounds halves to even, so an activation landing exactly on n+0.5 can
    // quantise one LSB apart.  Continuous activations make that measure-zero
    // but not impossible; validate with the differential validator before
    // deploying, since DSpark exactness depends on target numerics.
    q.x = __float2int_rn(xi.x*d_inv);
    q.y = __float2int_rn(xi.y*d_inv);
    q.z = __float2int_rn(xi.z*d_inv);
    q.w = __float2int_rn(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;
    }

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = d_inv > 0.0f ? 1.0f / d_inv : 0.0f;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = i4_grid ? rocmi4_w4a4_scale(amax)
                            : (d_inv > 0.0f ? 1.0f / d_inv : 0.0f);

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
#if GGML_ROCMI4_W4A4
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (type_src0 == GGML_TYPE_Q4_0_ROCMI4 && GGML_CUDA_CC_IS_GFX1151(cc)) {
        quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true>
            <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03,
                                                   ne0, ne1, ne2, 0, 0);
        return;
    }
#endif
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, 0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, 0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, 0);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_q8_1_grouped_cuda(
        const float * x, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t group_width,
        const int64_t token_stride, const int64_t group_stride,
        const int64_t ne0, const int64_t ne1, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(group_width > 0 && group_width % 4 == 0);
    GGML_ASSERT(ne00 % group_width == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    const int64_t block_num_y =
        (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
        (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
#if GGML_ROCMI4_W4A4
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (type_src0 == GGML_TYPE_Q4_0_ROCMI4 && GGML_CUDA_CC_IS_GFX1151(cc)) {
        quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true>
            <<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, vy, ne00, token_stride, 0, 0,
                ne0, ne1, 1, group_width, group_stride);
        return;
    }
#endif
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(
                    x, nullptr, vy, ne00, token_stride, 0, 0,
                    ne0, ne1, 1, group_width, group_stride);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(
                    x, nullptr, vy, ne00, token_stride, 0, 0,
                    ne0, ne1, 1, group_width, group_stride);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(
                    x, nullptr, vy, ne00, token_stride, 0, 0,
                    ne0, ne1, 1, group_width, group_stride);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
