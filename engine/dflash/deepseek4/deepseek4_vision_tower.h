// Native DeepSeek-V4-Flash-Vision-Exp tower execution.
//
// The mmproj remains operator-owned and lazy: constructing the language model
// does not open this file or reserve its device buffer. The first explicit
// tower load expands only the biased linears to F32 (so bias is applied before
// the BF16 output boundary) and keeps bias-free FFN matrices in BF16. Runtime
// inputs are already-resized BF16 patch rows produced by the pure preprocessing
// contract; image decoding and resizing deliberately remain outside this API.

#ifndef DFLASH_DEEPSEEK4_VISION_TOWER_H
#define DFLASH_DEEPSEEK4_VISION_TOWER_H

#include "deepseek4_vision_contract.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace dflash {

struct Deepseek4VisionTowerCheckpoints {
    int n_vit_h = 0;
    int n_vit_w = 0;
    int n_llm_h = 0;
    int n_llm_w = 0;
    // All checkpoint payloads are row-major F32. The first three have
    // n_vit_h*n_vit_w rows of width 1024; post_aligner has
    // n_llm_h*n_llm_w rows of width 4096.
    std::vector<float> post_patch_projection;
    std::vector<float> post_block_0;
    std::vector<float> post_vit;
    std::vector<float> post_aligner;
};

struct Deepseek4VisionBlock0Checkpoints {
    std::vector<float> norm1_out;
    std::vector<float> qkv_biased;
    std::vector<float> q_roped;
    std::vector<float> k_roped;
    std::vector<float> v_in;
    std::vector<float> sdpa_out;
    std::vector<float> wo_biased;
    std::vector<float> post_attn_residual;
    std::vector<float> norm2_out;
    std::vector<float> mlp_gate;
    std::vector<float> mlp_up;
    std::vector<float> mlp_silu_act;
    std::vector<float> mlp_down_out;
};

class Deepseek4VisionTower {
public:
    Deepseek4VisionTower();
    ~Deepseek4VisionTower();
    Deepseek4VisionTower(Deepseek4VisionTower && other) noexcept;
    Deepseek4VisionTower & operator=(Deepseek4VisionTower && other) noexcept;
    Deepseek4VisionTower(const Deepseek4VisionTower &) = delete;
    Deepseek4VisionTower & operator=(const Deepseek4VisionTower &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend bool deepseek4_load_vision_tower(
        const std::string &, int32_t, ggml_backend_t,
        Deepseek4VisionTower &, std::string &);
    friend bool deepseek4_run_vision_tower(
        Deepseek4VisionTower &, const std::vector<uint16_t> &, int, int,
        int32_t, int, Deepseek4VisionTowerCheckpoints &,
        Deepseek4VisionBlock0Checkpoints *, Deepseek4PreparedImage *,
        std::string &);
};

bool deepseek4_load_vision_tower(
    const std::string & mmproj_path, int32_t model_n_embd,
    ggml_backend_t backend, Deepseek4VisionTower & out, std::string & error);

// Execute one already-resized image. bf16_patches is [n_h*n_w, 588] in the
// exact row order emitted by deepseek4_vision_pack_rgb8_patches(). When
// prepared is non-null, the same post-aligner rows are assembled with the
// loaded marker weights through deepseek4_prepare_image().
bool deepseek4_run_vision_tower(
    Deepseek4VisionTower & tower,
    const std::vector<uint16_t> & bf16_patches,
    int n_vit_h, int n_vit_w,
    int32_t vocab_size, int start_pos,
    Deepseek4VisionTowerCheckpoints & checkpoints,
    Deepseek4VisionBlock0Checkpoints * block0_checkpoints,
    Deepseek4PreparedImage * prepared,
    std::string & error);

}  // namespace dflash

#endif  // DFLASH_DEEPSEEK4_VISION_TOWER_H
