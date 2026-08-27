// Qwen4Exp model identification and structural validation.
//
// Qwen3.8-Flash-Next declares `general.architecture = "qwen4exp"`.  It is
// not shape-compatible with Qwen3-Next or the DeepSeek4 backend: four-stream
// gated residuals, QSA, and the 320M-row PLE table all require dedicated
// runtime state. The dedicated backend now exposes correctness-first q=1 text
// generation; vision, MTP, batching, and constrained decoding stay fail-closed
// until their separate graph/state paths exist. Real-weight differential
// validation remains required before claiming token parity.
//
// Architecture/shape provenance:
//   Qwen/Qwen3.8-Flash-Next revision f5d08274bafd880402bd16f5e3e6c514136ec06c
//   transformers revision 36deb0b53ed0863f4b4dfdea23dcaec7f3df3701
// Experimental implementation references (not merged-support claims):
//   llama.cpp PR #27742 head 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
//   llama.cpp PR #27774 head abdc7a0bf815d3b83e26dd523c6960e4dd597e82
// PR #27774 adds optional KV-cache Hadamard rotations around QSA: apply
// self_k_rot to Q/K and self_v_rot to V before attention, then self_v_rot to
// the attention output. The q=1 runtime preserves that exact ordering whenever
// cache rotation is configured; its current F32 cache needs no rotation.

#pragma once

#include "qwen4exp_shards.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace dflash::common {

enum class ModelArchitecture {
    DEEPSEEK4,
    QWEN4_EXP,
    UNKNOWN,
};

inline ModelArchitecture model_architecture_from_name(const char * name) {
    if (!name) return ModelArchitecture::UNKNOWN;
    if (std::strcmp(name, "deepseek4") == 0) {
        return ModelArchitecture::DEEPSEEK4;
    }
    if (std::strcmp(name, "qwen4exp") == 0 ||
        std::strcmp(name, "qwen4_exp") == 0) {
        return ModelArchitecture::QWEN4_EXP;
    }
    return ModelArchitecture::UNKNOWN;
}

inline const char * model_architecture_name(ModelArchitecture architecture) {
    switch (architecture) {
        case ModelArchitecture::DEEPSEEK4: return "deepseek4";
        case ModelArchitecture::QWEN4_EXP: return "qwen4exp";
        case ModelArchitecture::UNKNOWN: return "unknown";
    }
    return "unknown";
}

struct Qwen4ExpContract {
    static constexpr uint32_t context_length = 262144;
    static constexpr uint32_t block_count = 48;
    static constexpr uint32_t embedding_length = 2560;
    static constexpr uint32_t vocab_size = 248320;
    static constexpr uint32_t qsa_interval = 4;
    static constexpr uint32_t ple_layer = 1;  // zero-based
    static constexpr uint32_t expert_count = 512;
    static constexpr uint32_t expert_used_count = 10;
    static constexpr uint32_t qsa_token_budget = 2048;
    static constexpr uint32_t qsa_block_top_k = 512;
};

inline bool qwen4exp_layer_is_qsa(uint32_t layer) {
    return layer < Qwen4ExpContract::block_count &&
           (layer + 1) % Qwen4ExpContract::qsa_interval == 0;
}

inline bool qwen4exp_layer_has_ple(uint32_t layer) {
    return layer == Qwen4ExpContract::ple_layer;
}

// Opens metadata only (no tensor allocation) and returns UNKNOWN on an
// unreadable/malformed file. `error` always explains UNKNOWN.
ModelArchitecture inspect_gguf_architecture(const char * model_path,
                                            std::string & error);

// Validate the released Qwen3.8-Flash-Next text checkpoint's exact scalar,
// array, tokenizer, and tensor-shape contract.  Quantized tensor types are
// intentionally unconstrained here; the later loader owns supported storage
// types.  This function allocates no weight buffers.
bool validate_qwen4exp_gguf(const char * model_path, std::string & error);

// Later speculative work must treat committed recurrent state as a separate
// frontier. HaloSpecKV commit 60ff854... is the accepted-prefix-replay
// reference requested for this port: verify state is transient; after the
// accepted length is known, replay exactly the accepted input prefix from the
// committed checkpoint using the same finite-precision recurrence, and update
// the width-4 causal-conv history from the same prefix. Rejected rows must
// never mutate committed state. This is documentation only, not an MTP port.

}  // namespace dflash::common
