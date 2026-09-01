// Versioned checkpoint oracle for the native Vision-Exp tower.
//
// The authority producer serializes both BF16-model and F32 diagnostic lanes
// as row-major F32 checkpoints. This parser keeps lane provenance and payload
// checksums explicit. It also computes the producer's truncated SHA-256 patch
// digest so a graph result can never be compared against checkpoints from a
// different resized input.

#ifndef DFLASH_DEEPSEEK4_VISION_ORACLE_H
#define DFLASH_DEEPSEEK4_VISION_ORACLE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash {

enum class Deepseek4VisionOracleLane : uint32_t {
    BF16 = 0,
    F32 = 1,
};

struct Deepseek4VisionOracleRecord {
    std::string name;
    Deepseek4VisionOracleLane lane = Deepseek4VisionOracleLane::BF16;
    std::vector<int64_t> shape;
    uint64_t checksum = 0;
    std::vector<float> values;
};

struct Deepseek4VisionOracle {
    uint32_t version = 0;
    std::string metadata_json;
    std::string image;
    std::string source_revision;
    std::string weight_digest;
    std::string patch_digest;
    Deepseek4VisionOracleLane lane = Deepseek4VisionOracleLane::BF16;
    int n_vit_h = 0;
    int n_vit_w = 0;
    int n_llm_h = 0;
    int n_llm_w = 0;
    int block = -1;
    std::vector<Deepseek4VisionOracleRecord> records;
};

struct Deepseek4VisionOracleComparison {
    double relative_l2 = 0.0;
    double cosine = 0.0;
    float max_abs = 0.0f;
    size_t max_abs_index = 0;
};

bool deepseek4_parse_vision_oracle(
    const std::vector<uint8_t> & bytes,
    Deepseek4VisionOracle & out, std::string & error);

bool deepseek4_load_vision_oracle(
    const std::string & path,
    Deepseek4VisionOracle & out, std::string & error);

bool deepseek4_parse_vision_block_oracle(
    const std::vector<uint8_t> & bytes,
    Deepseek4VisionOracle & out, std::string & error);

bool deepseek4_load_vision_block_oracle(
    const std::string & path,
    Deepseek4VisionOracle & out, std::string & error);

const Deepseek4VisionOracleRecord * deepseek4_vision_oracle_record(
    const Deepseek4VisionOracle & oracle, const std::string & name);

std::string deepseek4_vision_patch_digest(
    const std::vector<uint16_t> & bf16_patches);

bool deepseek4_compare_vision_checkpoint(
    const std::vector<float> & actual,
    const Deepseek4VisionOracleRecord & expected,
    Deepseek4VisionOracleComparison & out,
    std::string & error);

// The native-tower gate is relative to the authority producer's independently
// recorded effect of replacing its BF16 lane with F32. Keeping this predicate
// beside the comparison makes both the pass and a reachable red control
// GPU-free and reviewable before any engine output exists.
bool deepseek4_vision_ratio_gate(
    const Deepseek4VisionOracleComparison & engine_vs_bf16,
    const Deepseek4VisionOracleComparison & f32_vs_bf16,
    double maximum_ratio, double & ratio);

// A compounding budget is measured by carrying a seeded perturbation through
// the reference's own BF16 chain. Unlike the legacy per-record lane ratio, the
// resulting bound cannot tighten after a correct rounded boundary.
bool deepseek4_vision_budget_gate(
    const Deepseek4VisionOracleComparison & engine_vs_bf16,
    double maximum_relative_l2, double & consumption);

}  // namespace dflash

#endif  // DFLASH_DEEPSEEK4_VISION_ORACLE_H
