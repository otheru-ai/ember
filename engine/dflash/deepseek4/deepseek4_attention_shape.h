#pragma once

#include <cstddef>
#include <limits>

// Host-only classification of the MLA attention graph shape.
//
// Keep these predicates in one pure helper so the q=1 post-restore prefill
// case is testable without constructing a ggml graph or requiring a GPU.  In
// particular, q=1 sparse prefill is not ordinary decode: it has no cached
// graph inputs and therefore falls through to the dynamic raw-KV row writer.

namespace dflash::common {

struct DeepSeek4AttentionShape {
    bool causal_batch;
    bool layer_major_batch;
    bool fused_causal;
    int  first_raw_kv_write;
};

struct DeepSeek4ViewBound {
    bool        valid;
    std::size_t absolute_offs;
};

// Arithmetic half of ggml_new_tensor_impl's view invariant, split out so the
// process-preserving guard is covered by GPU-free tests. Subtraction after the
// offset comparison avoids overflow in `requested + absolute_offs`.
inline DeepSeek4ViewBound deepseek4_view_bound(
        std::size_t relative_offs,
        std::size_t source_view_offs,
        std::size_t requested_bytes,
        std::size_t base_nbytes) {
    if (relative_offs >
        std::numeric_limits<std::size_t>::max() - source_view_offs) {
        return {false, 0};
    }
    const std::size_t absolute_offs = relative_offs + source_view_offs;
    return {
        requested_bytes == 0 ||
            (absolute_offs <= base_nbytes &&
             requested_bytes <= base_nbytes - absolute_offs),
        absolute_offs,
    };
}

// Shared-memory footprint of the decode flash-attention kernel, in bytes.
// `ds4_flash_attn_d512_shared_kv_kernel` (ggml-cuda/fattn.cu) stages one float
// score per KV row plus two floats per compressed block, and its grouped
// variants only ever ask for more. Decode never fuses the inverse RoPE, so the
// kernel's 64-float rope tail is not allocated.
inline std::size_t deepseek4_decode_flash_shmem_bytes(
        int n_attn,
        int n_comp_attn,
        int comp_block_size) {
    if (n_attn <= 0 || comp_block_size <= 0) {
        return 0;
    }
    const std::size_t comp_blocks = n_comp_attn > 0
        ? (std::size_t) ((n_comp_attn + comp_block_size - 1) / comp_block_size)
        : 0;
    return ((std::size_t) n_attn + 2 * comp_blocks) * sizeof(float);
}

// q=1 eligibility for the exact D=512 flash kernel that already serves prefill.
//
// Only the fused decode graph's masked full-ring path qualifies. There the KV
// operand is the whole physical ring in slot order and the host-filled row mask
// is the sole visibility authority, so the kernel never has to infer
// chronological order — and its visible raw rows are the single contiguous span
// [0, valid) that the value-envelope scan expects. Every other q=1 shape builds
// chronological views of a partial ring with no mask at all, which is exactly
// the ordering assumption this kernel does not make.
//
// The kernel is D=512-only, and a long compressed span can push its score
// staging past the per-workgroup LDS budget. Both inputs here are functions of
// the fused decode graph's shape key, so the answer is stable for the lifetime
// of a cached graph.
// The WMMA decode kernel (ds4_decode_attn_wmma_partial) keeps all query heads
// resident and streams the KV span in fixed tiles, so its shared memory is a
// constant ~19.2 KB independent of n_attn -- unlike the dense kernels, whose
// staging grows with the span. When that kernel is applicable the span-based
// budget check does not apply and must not be allowed to refuse the very
// contexts where it wins most (at --max-ctx 131072 the dense budget would deny
// it above roughly 61k tokens).
//
// Applicability must mirror the dispatch in ggml_cuda_ds4_flash_attn_d512_f32:
// D=512 with exactly 64 query heads. It cannot check the runtime arch from
// here, so a non-RDNA3+ HIP device with this head count would fall through to
// the dense kernels and hit their budget. Ember targets gfx1151 only; revisit
// this coupling if that changes.
inline bool deepseek4_decode_wmma_applicable(int head_dim, int n_head) {
    return head_dim == 512 && n_head == 64;
}

inline bool deepseek4_decode_flash_eligible(
        int n_tokens,
        int head_dim,
        int n_attn,
        int n_comp_attn,
        bool masked_kv,
        int comp_block_size,
        std::size_t shmem_budget_bytes,
        int n_head = 0) {
    if (n_tokens != 1 || !masked_kv || head_dim != 512 || n_attn <= 0) {
        return false;
    }
    if (deepseek4_decode_wmma_applicable(head_dim, n_head)) {
        return true;   // constant shared memory; span budget does not bind
    }
    return deepseek4_decode_flash_shmem_bytes(
               n_attn, n_comp_attn, comp_block_size) <= shmem_budget_bytes;
}

inline DeepSeek4AttentionShape deepseek4_attention_shape(
        int n_tokens,
        int n_swa,
        bool has_cached_inputs,
        bool has_f32_array_inputs,
        bool causal_verify_enabled,
        bool optimized_attention,
        bool has_attn_row_mask) {
    const bool causal_batch =
        n_tokens > 1 && !has_cached_inputs && has_f32_array_inputs &&
        causal_verify_enabled;
    return {
        causal_batch,
        causal_batch && optimized_attention,
        n_tokens > 1 && has_cached_inputs && has_attn_row_mask,
        n_tokens > n_swa ? n_tokens - n_swa : 0,
    };
}

} // namespace dflash::common
