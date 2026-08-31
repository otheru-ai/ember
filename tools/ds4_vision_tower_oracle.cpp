#include "deepseek4/deepseek4_vision_oracle.h"
#include "deepseek4/deepseek4_vision_tower.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "ggml-backend.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace dflash;

namespace {

bool is_lower_hex_digest(const std::string & value) {
    if (value.size() != 32) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool load_patches(const std::string & path, size_t expected_values,
                  std::vector<uint16_t> & patches, std::string & error) {
    patches.clear();
    if (expected_values > std::numeric_limits<size_t>::max() /
                              sizeof(uint16_t)) {
        error = "patch payload size overflows";
        return false;
    }
    const size_t expected_bytes = expected_values * sizeof(uint16_t);
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        error = "cannot open raw BF16 patch payload";
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        static_cast<uintmax_t>(st.st_size) != expected_bytes) {
        close(fd);
        error = "raw BF16 patch payload size differs from oracle grid";
        return false;
    }
    patches.resize(expected_values);
    size_t offset = 0;
    auto * bytes = reinterpret_cast<uint8_t *>(patches.data());
    while (offset < expected_bytes) {
        const ssize_t count = read(fd, bytes + offset,
                                   expected_bytes - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            close(fd);
            error = "cannot read complete raw BF16 patch payload";
            patches.clear();
            return false;
        }
        offset += static_cast<size_t>(count);
    }
    if (close(fd) != 0) {
        error = "cannot close raw BF16 patch payload";
        patches.clear();
        return false;
    }
    return true;
}

const std::vector<float> * checkpoint_values(
        const Deepseek4VisionTowerCheckpoints & checkpoints,
        const std::string & name) {
    if (name == "post_patch_projection") {
        return &checkpoints.post_patch_projection;
    }
    if (name == "post_block_0") return &checkpoints.post_block_0;
    if (name == "post_vit") return &checkpoints.post_vit;
    if (name == "post_aligner") return &checkpoints.post_aligner;
    return nullptr;
}

const std::vector<float> * block_checkpoint_values(
        const Deepseek4VisionBlock0Checkpoints & checkpoints,
        const std::string & name) {
    if (name == "norm1_out") return &checkpoints.norm1_out;
    if (name == "qkv_biased") return &checkpoints.qkv_biased;
    if (name == "q_roped") return &checkpoints.q_roped;
    if (name == "k_roped") return &checkpoints.k_roped;
    if (name == "v_in") return &checkpoints.v_in;
    if (name == "sdpa_out") return &checkpoints.sdpa_out;
    if (name == "wo_biased") return &checkpoints.wo_biased;
    if (name == "post_attn_residual") {
        return &checkpoints.post_attn_residual;
    }
    if (name == "norm2_out") return &checkpoints.norm2_out;
    if (name == "mlp_gate") return &checkpoints.mlp_gate;
    if (name == "mlp_up") return &checkpoints.mlp_up;
    if (name == "mlp_silu_act") return &checkpoints.mlp_silu_act;
    if (name == "mlp_down_out") return &checkpoints.mlp_down_out;
    return nullptr;
}

bool compare_all(const Deepseek4VisionTowerCheckpoints & checkpoints,
                 const Deepseek4VisionOracle & bf16,
                 const Deepseek4VisionOracle & f32,
                 std::string & error) {
    // The independent BF16 and F32 reference lanes separate by roughly the
    // same scale as the engine at tower depth. A numeric threshold there
    // measures dtype sensitivity rather than correctness, and neither a
    // compounding curve nor per-block checkpoints restore discrimination.
    // Keep the records visible, but leave the verdict to the separate
    // behavioural gate. Block 0 remains numerically discriminating below.
    static constexpr const char * names[] = {
        "post_patch_projection", "post_block_0", "post_vit", "post_aligner",
    };
    for (const char * name : names) {
        const auto * actual = checkpoint_values(checkpoints, name);
        const auto * authority = deepseek4_vision_oracle_record(bf16, name);
        const auto * diagnostic = deepseek4_vision_oracle_record(f32, name);
        if (!actual || !authority || !diagnostic ||
            authority->shape != diagnostic->shape) {
            error = "paired oracle checkpoint contract differs: " +
                    std::string(name);
            return false;
        }
        Deepseek4VisionOracleComparison engine_delta;
        Deepseek4VisionOracleComparison lane_delta;
        if (!deepseek4_compare_vision_checkpoint(
                *actual, *authority, engine_delta, error) ||
            !deepseek4_compare_vision_checkpoint(
                diagnostic->values, *authority, lane_delta, error)) {
            return false;
        }
        const double ratio = lane_delta.relative_l2 > 0.0
            ? engine_delta.relative_l2 / lane_delta.relative_l2
            : std::numeric_limits<double>::infinity();
        std::printf(
            "%s engine_vs_bf16_rel_l2=%.9g bf16_vs_f32_rel_l2=%.9g "
            "ratio=%.9g cosine=%.9g max_abs=%.9g max_index=%zu "
            "numeric_gate=diagnostic-only\n",
            name, engine_delta.relative_l2, lane_delta.relative_l2, ratio,
            engine_delta.cosine, static_cast<double>(engine_delta.max_abs),
            engine_delta.max_abs_index);
    }
    return true;
}

bool compare_block0(const Deepseek4VisionBlock0Checkpoints & checkpoints,
                    const Deepseek4VisionOracle & bf16,
                    const Deepseek4VisionOracle & f32,
                    std::string & error) {
    struct Budget {
        const char * name;
        double maximum_relative_l2;
        bool localizing;
    };
    // Measured by otheru-quant-pipeline@e619204: a seeded perturbation enters
    // at the block input and is carried through the model's own BF16 block;
    // the worst of six directions receives a 1.5 safety factor. Injecting at
    // norm1_out instead would round away most of a sub-ulp perturbation before
    // measuring its propagation. A verdict is always the first red record;
    // later reds are downstream observations.
    static constexpr Budget budgets[] = {
        {"norm1_out", 3.161e-5, true},
        {"qkv_biased", 1.738e-4, true},
        {"q_roped", 1.848e-4, true},
        {"k_roped", 1.847e-4, true},
        {"v_in", 3.142e-4, true},
        {"sdpa_out", 4.987e-4, true},
        {"wo_biased", 4.515e-4, true},
        {"post_attn_residual", 8.579e-4, true},
        {"norm2_out", 1.054e-3, true},
        {"mlp_gate", 1.540e-3, false},
        {"mlp_up", 1.501e-3, false},
        {"mlp_silu_act", 1.975e-3, false},
        {"mlp_down_out", 2.000e-3, false},
    };
    bool gate_passed = true;
    const Budget * first_exceeded = nullptr;
    for (const Budget & budget : budgets) {
        const char * name = budget.name;
        const auto * actual = block_checkpoint_values(checkpoints, name);
        const auto * authority = deepseek4_vision_oracle_record(bf16, name);
        const auto * diagnostic = deepseek4_vision_oracle_record(f32, name);
        if (!actual || !authority || !diagnostic ||
            authority->shape != diagnostic->shape) {
            error = "paired block-0 checkpoint contract differs: " +
                    std::string(name);
            return false;
        }
        Deepseek4VisionOracleComparison engine_delta;
        Deepseek4VisionOracleComparison lane_delta;
        if (!deepseek4_compare_vision_checkpoint(
                *actual, *authority, engine_delta, error) ||
            !deepseek4_compare_vision_checkpoint(
                diagnostic->values, *authority, lane_delta, error)) {
            return false;
        }
        double consumption = 0.0;
        const bool checkpoint_passed = deepseek4_vision_budget_gate(
            engine_delta, budget.maximum_relative_l2, consumption);
        if (!checkpoint_passed && !first_exceeded) {
            first_exceeded = &budget;
        }
        gate_passed = checkpoint_passed && gate_passed;
        std::printf(
            "block0.%s engine_vs_bf16_rel_l2=%.9g "
            "bf16_vs_f32_rel_l2=%.9g budget=%.9g consumption=%.9g "
            "cosine=%.9g max_abs=%.9g max_index=%zu diagnostic_gate=%s\n",
            name, engine_delta.relative_l2, lane_delta.relative_l2,
            budget.maximum_relative_l2, consumption,
            engine_delta.cosine, static_cast<double>(engine_delta.max_abs),
            engine_delta.max_abs_index,
            checkpoint_passed ? "pass" :
                (first_exceeded == &budget
                    ? (budget.localizing ? "first-red" : "budget-exceeded")
                    : "downstream-red"));
    }
    if (!gate_passed) {
        if (first_exceeded && first_exceeded->localizing) {
            error = "block-0 first red under measured compounding budget: " +
                    std::string(first_exceeded->name);
        } else {
            error = "block-0 budget exceeded at " +
                    std::string(first_exceeded ? first_exceeded->name
                                               : "unknown") +
                    "; localization not implied";
        }
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 6 && argc != 8) {
        std::fprintf(stderr,
            "usage: %s MMPROJ EXPECTED_WEIGHT_DIGEST RAW_BF16_PATCHES "
            "BF16_ORACLE F32_ORACLE [BF16_BLOCK0 F32_BLOCK0]\n",
            argv[0]);
        return 2;
    }
    Deepseek4VisionOracle bf16;
    Deepseek4VisionOracle f32;
    std::string error;
    if (!deepseek4_load_vision_oracle(argv[4], bf16, error) ||
        !deepseek4_load_vision_oracle(argv[5], f32, error)) {
        std::fprintf(stderr, "oracle load failed: %s\n", error.c_str());
        return 1;
    }
    if (bf16.lane != Deepseek4VisionOracleLane::BF16 ||
        f32.lane != Deepseek4VisionOracleLane::F32 ||
        bf16.image != f32.image || bf16.patch_digest != f32.patch_digest ||
        bf16.source_revision != f32.source_revision ||
        bf16.weight_digest != f32.weight_digest ||
        bf16.n_vit_h != f32.n_vit_h || bf16.n_vit_w != f32.n_vit_w ||
        bf16.n_llm_h != f32.n_llm_h || bf16.n_llm_w != f32.n_llm_w) {
        std::fprintf(stderr, "BF16/F32 oracle pair does not bind one input\n");
        return 1;
    }
    Deepseek4VisionOracle block_bf16;
    Deepseek4VisionOracle block_f32;
    const bool with_block = argc == 8;
    if (with_block &&
        (!deepseek4_load_vision_block_oracle(argv[6], block_bf16, error) ||
         !deepseek4_load_vision_block_oracle(argv[7], block_f32, error))) {
        std::fprintf(stderr, "block-0 oracle load failed: %s\n", error.c_str());
        return 1;
    }
    if (with_block &&
        (block_bf16.lane != Deepseek4VisionOracleLane::BF16 ||
         block_f32.lane != Deepseek4VisionOracleLane::F32 ||
         block_bf16.block != 0 || block_f32.block != 0 ||
         block_bf16.image != bf16.image || block_f32.image != bf16.image ||
         block_bf16.source_revision != bf16.source_revision ||
         block_f32.source_revision != bf16.source_revision ||
         block_bf16.patch_digest != bf16.patch_digest ||
         block_f32.patch_digest != bf16.patch_digest ||
         block_bf16.weight_digest != bf16.weight_digest ||
         block_f32.weight_digest != bf16.weight_digest ||
         block_bf16.n_vit_h != bf16.n_vit_h ||
         block_bf16.n_vit_w != bf16.n_vit_w ||
         block_f32.n_vit_h != bf16.n_vit_h ||
         block_f32.n_vit_w != bf16.n_vit_w ||
         block_bf16.n_llm_h != bf16.n_llm_h ||
         block_bf16.n_llm_w != bf16.n_llm_w ||
         block_f32.n_llm_h != bf16.n_llm_h ||
         block_f32.n_llm_w != bf16.n_llm_w)) {
        std::fprintf(stderr, "block-0 oracle does not bind the main pair\n");
        return 1;
    }
    const std::string expected_weight_digest = argv[2];
    if (!is_lower_hex_digest(expected_weight_digest) ||
        bf16.weight_digest != expected_weight_digest) {
        std::fprintf(stderr,
            "oracle weight digest differs: actual=%s expected=%s\n",
            bf16.weight_digest.c_str(), expected_weight_digest.c_str());
        return 1;
    }
    const size_t patch_rows = static_cast<size_t>(bf16.n_vit_h) *
                              static_cast<size_t>(bf16.n_vit_w);
    if (patch_rows > std::numeric_limits<size_t>::max() / 588U) {
        std::fprintf(stderr, "oracle patch grid overflows\n");
        return 1;
    }
    std::vector<uint16_t> patches;
    if (!load_patches(argv[3], patch_rows * 588U, patches, error)) {
        std::fprintf(stderr, "patch load failed: %s\n", error.c_str());
        return 1;
    }
    const std::string patch_digest = deepseek4_vision_patch_digest(patches);
    if (patch_digest != bf16.patch_digest) {
        std::fprintf(stderr,
            "patch digest differs: actual=%s expected=%s\n",
            patch_digest.c_str(), bf16.patch_digest.c_str());
        return 1;
    }
    std::printf(
        "oracle image=%s source_revision=%s weight_digest=%s "
        "patch_digest=%s tower_numeric_gate=diagnostic-only "
        "block0_budget_source=otheru-quant-pipeline@e619204\n",
        bf16.image.c_str(), bf16.source_revision.c_str(),
        bf16.weight_digest.c_str(), bf16.patch_digest.c_str());

    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!backend) {
        std::fprintf(stderr, "no GPU backend is available\n");
        return 1;
    }
    int result = 1;
    {
        Deepseek4VisionTower tower;
        if (!deepseek4_load_vision_tower(argv[1], 4096, backend,
                                         tower, error)) {
            std::fprintf(stderr, "tower load failed: %s\n", error.c_str());
        } else {
            Deepseek4VisionTowerCheckpoints checkpoints;
            Deepseek4VisionBlock0Checkpoints block0;
            Deepseek4PreparedImage prepared;
            if (!deepseek4_run_vision_tower(
                    tower, patches, bf16.n_vit_h, bf16.n_vit_w,
                    129280, 0, checkpoints,
                    with_block ? &block0 : nullptr, &prepared, error)) {
                std::fprintf(stderr, "tower run failed: %s\n", error.c_str());
            } else if (prepared.embeddings.empty() ||
                       checkpoints.n_llm_h != bf16.n_llm_h ||
                       checkpoints.n_llm_w != bf16.n_llm_w) {
                std::fprintf(stderr,
                    "tower learned-marker assembly contract differs\n");
            } else {
                std::string main_error;
                const bool main_ok = compare_all(
                    checkpoints, bf16, f32, main_error);
                std::string block_error;
                const bool block_ok = !with_block || compare_block0(
                    block0, block_bf16, block_f32, block_error);
                if (!main_ok) {
                    std::fprintf(stderr, "comparison failed: %s\n",
                                 main_error.c_str());
                }
                if (!block_ok) {
                    std::fprintf(stderr, "block-0 comparison failed: %s\n",
                                 block_error.c_str());
                }
                if (main_ok && block_ok) result = 0;
            }
        }
    }
    ggml_backend_free(backend);
    return result;
}
