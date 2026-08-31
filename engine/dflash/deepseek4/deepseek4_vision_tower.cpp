#include "deepseek4_vision_tower.h"

#include "deepseek4_vision_mmproj.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <unordered_map>
#include <utility>
#include <unistd.h>

namespace dflash {
namespace {

constexpr size_t kWeightContextBytes = 8U * 1024U * 1024U;
constexpr size_t kGraphContextBytes = 64U * 1024U * 1024U;
constexpr int kGraphNodes = 8192;

// Reference representation versus backend interface (vision.py at 47bede8):
//
// stage                    represented value       backend-facing value
// patch / qkv / residual   BF16                    BF16 widened for F32 ops
// RMSNorm                  F32 math -> BF16         F32 norm/mul, then BF16
// 2D RoPE Q/K              F32 math -> BF16         F32 rope, then BF16
// attention Q/K/V          rounded BF16             exact widening to F32
// attention result         BF16 after SDPA           F32 result, then BF16
// SwiGLU / residual adds   BF16 after every op       F32 op, then BF16
// post-ViT pad/unfold      rounded BF16             exact widening to F32
// biased aligner linears   BF16 after bias           F32 op, then BF16
//
// The stored F16 weights originate from BF16, but F16's smaller exponent range
// means that conversion is not universally lossless for the tiniest values.
// The loader widens the stored values without another loss. Biased weights are
// widened to F32 so bias remains inside the accumulator before the BF16 output
// boundary; bias-free FFN weights stay BF16. An interface widening never moves
// a rounding boundary or authorizes subsequent F32-only semantic stages.

bool read_exact_at(int fd, size_t offset, void * destination, size_t bytes) {
    auto * out = static_cast<uint8_t *>(destination);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(
            fd, out + done, bytes - done,
            static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        done += static_cast<size_t>(n);
    }
    return true;
}

bool is_bias_free_ffn_weight(const std::string & name) {
    return name.find(".ffn_gate.weight") != std::string::npos ||
           name.find(".ffn_up.weight") != std::string::npos ||
           name.find(".ffn_down.weight") != std::string::npos;
}

ggml_type runtime_type(const Deepseek4VisionTensorSlice & slice) {
    if (slice.storage == Deepseek4VisionStorage::F32) return GGML_TYPE_F32;
    return is_bias_free_ffn_weight(slice.name)
        ? GGML_TYPE_BF16 : GGML_TYPE_F32;
}

bool checked_product(int a, int b, int & result) {
    if (a <= 0 || b <= 0 || a > INT32_MAX / b) return false;
    result = a * b;
    return true;
}

ggml_tensor * cast_to(ggml_context * ctx, ggml_tensor * value,
                      ggml_type type) {
    return value->type == type ? value : ggml_cast(ctx, value, type);
}

ggml_tensor * round_bf16(ggml_context * ctx, ggml_tensor * value) {
    return cast_to(ctx, value, GGML_TYPE_BF16);
}

ggml_tensor * add_bf16(ggml_context * ctx, ggml_tensor * left,
                       ggml_tensor * right) {
    return round_bf16(ctx, ggml_add(
        ctx, cast_to(ctx, left, GGML_TYPE_F32),
        cast_to(ctx, right, GGML_TYPE_F32)));
}

ggml_tensor * rms_bf16(ggml_context * ctx, ggml_tensor * value,
                       ggml_tensor * weight, float epsilon) {
    ggml_tensor * normalized = ggml_rms_norm(
        ctx, cast_to(ctx, value, GGML_TYPE_F32), epsilon);
    return round_bf16(ctx, ggml_mul(ctx, normalized, weight));
}

ggml_tensor * linear_biased_bf16(
        ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
        ggml_tensor * value) {
    ggml_tensor * projected = ggml_mul_mat(
        ctx, weight, cast_to(ctx, value, GGML_TYPE_F32));
    ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
    return round_bf16(ctx, ggml_add(ctx, projected, bias));
}

ggml_tensor * linear_unbiased_bf16(
        ggml_context * ctx, ggml_tensor * weight, ggml_tensor * value) {
    ggml_tensor * projected = ggml_mul_mat(ctx, weight, value);
    ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
    return round_bf16(ctx, projected);
}

bool tensor_to_f32(ggml_tensor * tensor, std::vector<float> & output) {
    if (!tensor || tensor->type != GGML_TYPE_F32) return false;
    const int64_t count = ggml_nelements(tensor);
    if (count < 0 || static_cast<uint64_t>(count) >
            static_cast<uint64_t>(SIZE_MAX / sizeof(float))) {
        return false;
    }
    output.resize(static_cast<size_t>(count));
    ggml_backend_tensor_get(
        tensor, output.data(), 0, output.size() * sizeof(float));
    return true;
}

void set_name(ggml_tensor * tensor, const char * name) {
    if (tensor) ggml_set_name(tensor, name);
}

}  // namespace

struct Deepseek4VisionTower::Impl {
    ggml_backend_t backend = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    Deepseek4VisionMmprojMetadata metadata;
    std::unordered_map<std::string, ggml_tensor *> weights;
    Deepseek4ImageMarkers markers;

    ~Impl() {
        if (weights_ctx) ggml_free(weights_ctx);
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
    }

    ggml_tensor * get(const std::string & name) const {
        const auto found = weights.find(name);
        return found == weights.end() ? nullptr : found->second;
    }
};

Deepseek4VisionTower::Deepseek4VisionTower() = default;
Deepseek4VisionTower::~Deepseek4VisionTower() = default;
Deepseek4VisionTower::Deepseek4VisionTower(
    Deepseek4VisionTower && other) noexcept = default;
Deepseek4VisionTower & Deepseek4VisionTower::operator=(
    Deepseek4VisionTower && other) noexcept = default;

bool deepseek4_load_vision_tower(
        const std::string & mmproj_path, int32_t model_n_embd,
        ggml_backend_t backend, Deepseek4VisionTower & out,
        std::string & error) {
    error.clear();
    if (!backend) {
        error = "DeepSeek4 vision tower requires a backend";
        return false;
    }
    Deepseek4VisionMmprojMetadata metadata;
    if (!deepseek4_load_vision_mmproj_metadata(
            mmproj_path, model_n_embd, metadata, error)) {
        return false;
    }
    auto impl = std::make_unique<Deepseek4VisionTower::Impl>();
    impl->backend = backend;
    impl->metadata = metadata;
    ggml_init_params params {
        /*.mem_size=*/kWeightContextBytes,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/true,
    };
    impl->weights_ctx = ggml_init(params);
    if (!impl->weights_ctx) {
        error = "cannot create DeepSeek4 vision weight context";
        return false;
    }
    impl->weights.reserve(metadata.tensors.size());
    for (const auto & slice : metadata.tensors) {
        ggml_tensor * tensor = ggml_new_tensor(
            impl->weights_ctx, runtime_type(slice),
            static_cast<int>(slice.shape.size()), slice.shape.data());
        if (!tensor) {
            error = "cannot create DeepSeek4 vision tensor: " + slice.name;
            return false;
        }
        ggml_set_name(tensor, slice.name.c_str());
        impl->weights.emplace(slice.name, tensor);
    }
    impl->weights_buffer = ggml_backend_alloc_ctx_tensors(
        impl->weights_ctx, backend);
    if (!impl->weights_buffer) {
        error = "cannot allocate DeepSeek4 vision weight buffer";
        return false;
    }
    ggml_backend_buffer_set_usage(
        impl->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const int fd = open(mmproj_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        error = "cannot reopen DeepSeek4 mmproj payload";
        return false;
    }
    bool loaded = true;
    for (const auto & slice : metadata.tensors) {
        ggml_tensor * destination = impl->get(slice.name);
        if (!destination) {
            error = "DeepSeek4 vision tensor binding disappeared: " +
                    slice.name;
            loaded = false;
            break;
        }
        if (slice.storage == Deepseek4VisionStorage::F32) {
            std::vector<float> values(slice.byte_count / sizeof(float));
            if (!read_exact_at(fd, slice.file_offset, values.data(),
                               slice.byte_count)) {
                error = "cannot read DeepSeek4 vision tensor: " + slice.name;
                loaded = false;
                break;
            }
            ggml_backend_tensor_set(
                destination, values.data(), 0, slice.byte_count);
            if (slice.name == "mm.image_begin.weight") {
                impl->markers.start = values;
            } else if (slice.name == "mm.image_pad.weight") {
                impl->markers.pad = values;
            } else if (slice.name == "v.image_newline.weight") {
                impl->markers.newline = values;
            } else if (slice.name == "mm.image_end.weight") {
                impl->markers.end = values;
            }
            continue;
        }

        const size_t count = slice.byte_count / sizeof(ggml_fp16_t);
        std::vector<ggml_fp16_t> f16(count);
        if (!read_exact_at(fd, slice.file_offset, f16.data(),
                           slice.byte_count)) {
            error = "cannot read DeepSeek4 vision tensor: " + slice.name;
            loaded = false;
            break;
        }
        std::vector<float> f32(count);
        ggml_fp16_to_fp32_row(
            f16.data(), f32.data(), static_cast<int64_t>(count));
        if (destination->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(
                destination, f32.data(), 0, f32.size() * sizeof(float));
        } else {
            std::vector<ggml_bf16_t> bf16(count);
            ggml_fp32_to_bf16_row(
                f32.data(), bf16.data(), static_cast<int64_t>(count));
            ggml_backend_tensor_set(
                destination, bf16.data(), 0,
                bf16.size() * sizeof(ggml_bf16_t));
        }
    }
    const int close_result = close(fd);
    if (!loaded || close_result != 0) {
        if (loaded) error = "cannot close DeepSeek4 mmproj payload";
        return false;
    }
    const size_t marker_width = static_cast<size_t>(model_n_embd);
    if (impl->markers.start.size() != marker_width ||
        impl->markers.pad.size() != marker_width ||
        impl->markers.newline.size() != marker_width ||
        impl->markers.end.size() != marker_width) {
        error = "DeepSeek4 vision marker payload is incomplete";
        return false;
    }
    ggml_backend_synchronize(backend);
    out.impl_ = std::move(impl);
    return true;
}

bool deepseek4_run_vision_tower(
        Deepseek4VisionTower & tower,
        const std::vector<uint16_t> & bf16_patches,
        int n_vit_h, int n_vit_w,
        int32_t vocab_size, int start_pos,
        Deepseek4VisionTowerCheckpoints & checkpoints,
        Deepseek4VisionBlock0Checkpoints * block0_checkpoints,
        Deepseek4PreparedImage * prepared,
        std::string & error) {
    checkpoints = {};
    if (block0_checkpoints) *block0_checkpoints = {};
    if (prepared) *prepared = {};
    error.clear();
    if (!tower.impl_ || !tower.impl_->backend) {
        error = "DeepSeek4 vision tower is not loaded";
        return false;
    }
    const auto & config = tower.impl_->metadata.config;
    int n_patches = 0;
    if (!checked_product(n_vit_h, n_vit_w, n_patches)) {
        error = "invalid DeepSeek4 vision patch grid";
        return false;
    }
    const size_t patch_width = static_cast<size_t>(
        3 * config.patch_size * config.patch_size);
    if (static_cast<size_t>(n_patches) >
            std::numeric_limits<size_t>::max() / patch_width ||
        bf16_patches.size() != static_cast<size_t>(n_patches) * patch_width) {
        error = "DeepSeek4 vision patch payload differs from its grid";
        return false;
    }
    const int n_llm_h = (n_vit_h + config.scale_factor - 1) /
                        config.scale_factor;
    const int n_llm_w = (n_vit_w + config.scale_factor - 1) /
                        config.scale_factor;
    int n_aligner_rows = 0;
    if (!checked_product(n_llm_h, n_llm_w, n_aligner_rows)) {
        error = "invalid DeepSeek4 vision aligner grid";
        return false;
    }

    ggml_init_params params {
        /*.mem_size=*/kGraphContextBytes,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        error = "cannot create DeepSeek4 vision graph context";
        return false;
    }
    ggml_gallocr_t allocator = nullptr;
    bool ok = false;
    int padded_h = 0;
    int padded_w = 0;
    std::vector<int32_t> position_values;
    std::vector<int32_t> shuffle_values;

    ggml_tensor * patches = ggml_new_tensor_2d(
        ctx, GGML_TYPE_BF16, static_cast<int64_t>(patch_width), n_patches);
    ggml_tensor * positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, static_cast<int64_t>(4) * n_patches);
    ggml_tensor * shuffle_indices = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32,
        static_cast<int64_t>(config.scale_factor * config.scale_factor) *
            n_aligner_rows);
    ggml_set_input(patches);
    ggml_set_input(positions);
    ggml_set_input(shuffle_indices);
    set_name(patches, "vision.patches");
    set_name(positions, "vision.positions");
    set_name(shuffle_indices, "vision.shuffle_indices");

    const auto weight = [&tower](const std::string & name) {
        return tower.impl_->get(name);
    };
    ggml_tensor * current = linear_biased_bf16(
        ctx, weight("v.patch_embd.weight"), weight("v.patch_embd.bias"),
        patches);
    ggml_tensor * post_patch = cast_to(ctx, current, GGML_TYPE_F32);
    set_name(post_patch, "vision.post_patch_projection");
    ggml_set_output(post_patch);

    const int head_dim = config.embedding_length / config.head_count;
    const size_t bf16_bytes = sizeof(ggml_bf16_t);
    int sections[GGML_MROPE_SECTIONS] = {head_dim / 4, head_dim / 4, 0, 0};
    ggml_tensor * post_block_0 = nullptr;
    ggml_tensor * block0_norm1 = nullptr;
    ggml_tensor * block0_qkv = nullptr;
    ggml_tensor * block0_q = nullptr;
    ggml_tensor * block0_k = nullptr;
    ggml_tensor * block0_v = nullptr;
    ggml_tensor * block0_sdpa = nullptr;
    ggml_tensor * block0_wo = nullptr;
    ggml_tensor * block0_post_attn = nullptr;
    ggml_tensor * block0_norm2 = nullptr;
    ggml_tensor * block0_gate = nullptr;
    ggml_tensor * block0_up = nullptr;
    ggml_tensor * block0_activated = nullptr;
    ggml_tensor * block0_down = nullptr;
    const auto capture = [ctx, block0_checkpoints](
            ggml_tensor * value, const char * name) -> ggml_tensor * {
        if (!block0_checkpoints) return nullptr;
        ggml_tensor * output = cast_to(ctx, value, GGML_TYPE_F32);
        set_name(output, name);
        ggml_set_output(output);
        return output;
    };
    for (int layer = 0; layer < config.block_count; ++layer) {
        const std::string prefix = "v.blk." + std::to_string(layer) + ".";
        ggml_tensor * residual = current;
        ggml_tensor * normalized = rms_bf16(
            ctx, current, weight(prefix + "ln1.weight"),
            config.norm_epsilon);
        if (layer == 0) {
            block0_norm1 = capture(normalized, "vision.block0.norm1_out");
        }
        ggml_tensor * qkv = linear_biased_bf16(
            ctx, weight(prefix + "attn_qkv.weight"),
            weight(prefix + "attn_qkv.bias"), normalized);
        if (layer == 0) {
            block0_qkv = capture(qkv, "vision.block0.qkv_biased");
        }
        ggml_tensor * query = ggml_view_3d(
            ctx, qkv, head_dim, config.head_count, n_patches,
            static_cast<size_t>(head_dim) * bf16_bytes, qkv->nb[1], 0);
        ggml_tensor * key = ggml_view_3d(
            ctx, qkv, head_dim, config.head_count, n_patches,
            static_cast<size_t>(head_dim) * bf16_bytes, qkv->nb[1],
            static_cast<size_t>(config.embedding_length) * bf16_bytes);
        ggml_tensor * value = ggml_view_3d(
            ctx, qkv, head_dim, config.head_count, n_patches,
            static_cast<size_t>(head_dim) * bf16_bytes, qkv->nb[1],
            static_cast<size_t>(2 * config.embedding_length) * bf16_bytes);
        query = ggml_rope_multi(
            ctx, cast_to(ctx, query, GGML_TYPE_F32), positions, nullptr,
            head_dim / 2, sections, GGML_ROPE_TYPE_VISION,
            n_patches, config.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        key = ggml_rope_multi(
            ctx, cast_to(ctx, key, GGML_TYPE_F32), positions, nullptr,
            head_dim / 2, sections, GGML_ROPE_TYPE_VISION,
            n_patches, config.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        ggml_tensor * query_bf16 = round_bf16(ctx, query);
        ggml_tensor * key_bf16 = round_bf16(ctx, key);
        if (layer == 0) {
            block0_q = capture(query_bf16, "vision.block0.q_roped");
            block0_k = capture(key_bf16, "vision.block0.k_roped");
            block0_v = capture(value, "vision.block0.v_in");
        }
        // vision.py:Attention.forward calls PyTorch SDPA on BF16 Q/K/V. Its
        // math definition accumulates the score, softmax, and value products
        // in F32, then returns BF16. Keep that precision boundary explicit:
        // widening these already-rounded values is lossless, while the final
        // round matches SDPA's output dtype. The fused ggml flash kernel is a
        // valid lower-precision implementation, but it exceeds the official
        // model oracle's independently measured BF16-vs-F32 error budget.
        query = ggml_cont(ctx, ggml_permute(
            ctx, cast_to(ctx, query_bf16, GGML_TYPE_F32),
            0, 2, 1, 3));
        key = ggml_cont(ctx, ggml_permute(
            ctx, cast_to(ctx, key_bf16, GGML_TYPE_F32), 0, 2, 1, 3));
        value = ggml_cont(ctx, ggml_permute(
            ctx, cast_to(ctx, value, GGML_TYPE_F32), 0, 2, 1, 3));
        const float attention_operand_scale = std::sqrt(
            1.0f / std::sqrt(static_cast<float>(head_dim)));
        query = ggml_scale(ctx, query, attention_operand_scale);
        key = ggml_scale(ctx, key, attention_operand_scale);
        ggml_tensor * scores = ggml_mul_mat(ctx, key, query);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor * probabilities = ggml_soft_max(ctx, scores);
        ggml_tensor * value_transposed = ggml_cont(
            ctx, ggml_transpose(ctx, value));
        ggml_tensor * attended = ggml_mul_mat(
            ctx, value_transposed, probabilities);
        ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
        attended = ggml_cont(ctx, ggml_permute(
            ctx, attended, 0, 2, 1, 3));
        attended = round_bf16(ctx, ggml_reshape_2d(
            ctx, attended, config.embedding_length, n_patches));
        if (layer == 0) {
            block0_sdpa = capture(attended, "vision.block0.sdpa_out");
        }
        ggml_tensor * attention_output = linear_biased_bf16(
            ctx, weight(prefix + "attn_out.weight"),
            weight(prefix + "attn_out.bias"), attended);
        if (layer == 0) {
            block0_wo = capture(attention_output, "vision.block0.wo_biased");
        }
        current = add_bf16(ctx, residual, attention_output);
        if (layer == 0) {
            block0_post_attn = capture(
                current, "vision.block0.post_attn_residual");
        }

        residual = current;
        normalized = rms_bf16(
            ctx, current, weight(prefix + "ln2.weight"),
            config.norm_epsilon);
        if (layer == 0) {
            block0_norm2 = capture(normalized, "vision.block0.norm2_out");
        }
        ggml_tensor * gate = linear_unbiased_bf16(
            ctx, weight(prefix + "ffn_gate.weight"), normalized);
        ggml_tensor * up = linear_unbiased_bf16(
            ctx, weight(prefix + "ffn_up.weight"), normalized);
        if (layer == 0) {
            block0_gate = capture(gate, "vision.block0.mlp_gate");
            block0_up = capture(up, "vision.block0.mlp_up");
        }
        gate = round_bf16(ctx, ggml_silu(
            ctx, cast_to(ctx, gate, GGML_TYPE_F32)));
        ggml_tensor * activated = round_bf16(ctx, ggml_mul(
            ctx, cast_to(ctx, gate, GGML_TYPE_F32),
            cast_to(ctx, up, GGML_TYPE_F32)));
        if (layer == 0) {
            block0_activated = capture(
                activated, "vision.block0.mlp_silu_act");
        }
        ggml_tensor * down = linear_unbiased_bf16(
            ctx, weight(prefix + "ffn_down.weight"), activated);
        if (layer == 0) {
            block0_down = capture(down, "vision.block0.mlp_down_out");
        }
        current = add_bf16(ctx, residual, down);
        if (layer == 0) {
            post_block_0 = cast_to(ctx, current, GGML_TYPE_F32);
            set_name(post_block_0, "vision.post_block_0");
            ggml_set_output(post_block_0);
        }
    }

    current = rms_bf16(
        ctx, current, weight("v.post_ln.weight"), config.norm_epsilon);
    ggml_tensor * post_vit = cast_to(ctx, current, GGML_TYPE_F32);
    set_name(post_vit, "vision.post_vit");
    ggml_set_output(post_vit);

    // ROCm pad consumes F32. post_vit is the exact widening of the already
    // BF16-rounded final RMSNorm output, so this does not move a boundary.
    ggml_tensor * grid = ggml_reshape_3d(
        ctx, post_vit, config.embedding_length, n_vit_w, n_vit_h);
    if (!deepseek4_vision_pixel_shuffle_indices(
            n_vit_h, n_vit_w, padded_h, padded_w,
            shuffle_values, &error)) {
        ggml_free(ctx);
        return false;
    }
    grid = ggml_pad(
        ctx, grid, 0, padded_w - n_vit_w, padded_h - n_vit_h, 0);
    grid = ggml_reshape_2d(
        ctx, ggml_cont(ctx, grid), config.embedding_length,
        static_cast<int64_t>(padded_h) * padded_w);
    ggml_tensor * gathered = ggml_get_rows(ctx, grid, shuffle_indices);
    gathered = ggml_reshape_3d(
        ctx, gathered, config.embedding_length,
        config.scale_factor * config.scale_factor, n_aligner_rows);
    // ne[0] is fastest in GGML: after this swap, each channel's nine local
    // positions are contiguous, matching F.unfold's c*9 + ky*3 + kx order.
    gathered = ggml_cont(ctx, ggml_permute(ctx, gathered, 1, 0, 2, 3));
    gathered = ggml_reshape_2d(
        ctx, gathered,
        static_cast<int64_t>(config.embedding_length) *
            config.scale_factor * config.scale_factor,
        n_aligner_rows);
    current = linear_biased_bf16(
        ctx, weight("mm.1.weight"), weight("mm.1.bias"), gathered);
    current = round_bf16(ctx, ggml_gelu_erf(
        ctx, cast_to(ctx, current, GGML_TYPE_F32)));
    current = linear_biased_bf16(
        ctx, weight("mm.2.weight"), weight("mm.2.bias"), current);
    ggml_tensor * post_aligner = cast_to(ctx, current, GGML_TYPE_F32);
    set_name(post_aligner, "vision.post_aligner");
    ggml_set_output(post_aligner);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kGraphNodes, false);
    ggml_build_forward_expand(graph, post_patch);
    if (post_block_0) ggml_build_forward_expand(graph, post_block_0);
    ggml_build_forward_expand(graph, post_vit);
    ggml_build_forward_expand(graph, post_aligner);
    if (block0_checkpoints) {
        ggml_tensor * block_outputs[] = {
            block0_norm1, block0_qkv, block0_q, block0_k, block0_v,
            block0_sdpa, block0_wo, block0_post_attn, block0_norm2,
            block0_gate, block0_up, block0_activated, block0_down,
        };
        for (ggml_tensor * output : block_outputs) {
            ggml_build_forward_expand(graph, output);
        }
    }
    allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(tower.impl_->backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        error = "cannot allocate DeepSeek4 vision compute graph";
        goto done;
    }

    ggml_backend_tensor_set(
        patches, bf16_patches.data(), 0,
        bf16_patches.size() * sizeof(uint16_t));
    position_values.assign(static_cast<size_t>(4) * n_patches, 0);
    for (int row = 0; row < n_vit_h; ++row) {
        for (int column = 0; column < n_vit_w; ++column) {
            const int index = row * n_vit_w + column;
            position_values[static_cast<size_t>(index)] = row;
            position_values[static_cast<size_t>(n_patches + index)] = column;
        }
    }
    ggml_backend_tensor_set(
        positions, position_values.data(), 0,
        position_values.size() * sizeof(int32_t));
    ggml_backend_tensor_set(
        shuffle_indices, shuffle_values.data(), 0,
        shuffle_values.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(tower.impl_->backend, graph) !=
        GGML_STATUS_SUCCESS) {
        error = "DeepSeek4 vision graph compute failed";
        goto done;
    }
    ggml_backend_synchronize(tower.impl_->backend);

    checkpoints.n_vit_h = n_vit_h;
    checkpoints.n_vit_w = n_vit_w;
    checkpoints.n_llm_h = n_llm_h;
    checkpoints.n_llm_w = n_llm_w;
    if (!tensor_to_f32(post_patch, checkpoints.post_patch_projection) ||
        !tensor_to_f32(post_block_0, checkpoints.post_block_0) ||
        !tensor_to_f32(post_vit, checkpoints.post_vit) ||
        !tensor_to_f32(post_aligner, checkpoints.post_aligner)) {
        error = "cannot read DeepSeek4 vision checkpoints";
        goto done;
    }
    if (block0_checkpoints &&
        (!tensor_to_f32(block0_norm1, block0_checkpoints->norm1_out) ||
         !tensor_to_f32(block0_qkv, block0_checkpoints->qkv_biased) ||
         !tensor_to_f32(block0_q, block0_checkpoints->q_roped) ||
         !tensor_to_f32(block0_k, block0_checkpoints->k_roped) ||
         !tensor_to_f32(block0_v, block0_checkpoints->v_in) ||
         !tensor_to_f32(block0_sdpa, block0_checkpoints->sdpa_out) ||
         !tensor_to_f32(block0_wo, block0_checkpoints->wo_biased) ||
         !tensor_to_f32(block0_post_attn,
                        block0_checkpoints->post_attn_residual) ||
         !tensor_to_f32(block0_norm2, block0_checkpoints->norm2_out) ||
         !tensor_to_f32(block0_gate, block0_checkpoints->mlp_gate) ||
         !tensor_to_f32(block0_up, block0_checkpoints->mlp_up) ||
         !tensor_to_f32(block0_activated,
                        block0_checkpoints->mlp_silu_act) ||
         !tensor_to_f32(block0_down, block0_checkpoints->mlp_down_out))) {
        error = "cannot read DeepSeek4 vision block-0 checkpoints";
        goto done;
    }
    if (prepared && !deepseek4_prepare_image(
            vocab_size, n_llm_h, n_llm_w, start_pos,
            config.output_embedding_length, checkpoints.post_aligner,
            tower.impl_->markers, *prepared, &error)) {
        goto done;
    }
    ok = true;

done:
    if (!ok) {
        checkpoints = {};
        if (block0_checkpoints) *block0_checkpoints = {};
        if (prepared) *prepared = {};
    }
    if (allocator) ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

}  // namespace dflash
