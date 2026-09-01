#include "dflash/deepseek4/deepseek4_vision_oracle.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash;

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++g_pass; }                                                \
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); }           \
} while (0)

namespace {

void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

uint64_t fnv1a(const std::vector<uint8_t> & payload) {
    uint64_t hash = 14695981039346656037ULL;
    for (uint8_t byte : payload) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void append_record_shape(std::vector<uint8_t> & bytes,
                         const std::string & name,
                         const std::vector<int64_t> & shape, float base) {
    append_u32(bytes, static_cast<uint32_t>(name.size()));
    bytes.insert(bytes.end(), name.begin(), name.end());
    append_u32(bytes, 0);
    append_u32(bytes, static_cast<uint32_t>(shape.size()));
    uint64_t count = 1;
    for (int64_t extent : shape) {
        append_u64(bytes, static_cast<uint64_t>(extent));
        count *= static_cast<uint64_t>(extent);
    }
    append_u64(bytes, count);
    std::vector<uint8_t> payload(static_cast<size_t>(count) * sizeof(float));
    for (size_t i = 0; i < static_cast<size_t>(count); ++i) {
        const float value = base + static_cast<float>(i % 17) * 0.001f;
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        payload[i * 4] = static_cast<uint8_t>(bits);
        payload[i * 4 + 1] = static_cast<uint8_t>(bits >> 8);
        payload[i * 4 + 2] = static_cast<uint8_t>(bits >> 16);
        payload[i * 4 + 3] = static_cast<uint8_t>(bits >> 24);
    }
    append_u64(bytes, fnv1a(payload));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
}

void append_record(std::vector<uint8_t> & bytes, const std::string & name,
                   int rows, int width, float base) {
    append_record_shape(bytes, name, {rows, width}, base);
}

std::vector<uint8_t> fixture() {
    const std::string metadata =
        "{\"image\": \"fixture.png\", \"lane\": \"bf16\", "
        "\"n_llm_h\": 1, \"n_llm_w\": 1, \"n_vit_h\": 1, "
        "\"n_vit_w\": 1, "
        "\"patch_digest\": \"0123456789abcdef0123456789abcdef\", "
        "\"source_revision\": \"47bede8\", "
        "\"weight_digest\": \"abcdef0123456789abcdef0123456789\"}";
    std::vector<uint8_t> bytes {'D','S','4','O','R','C',0,0};
    append_u32(bytes, 1);
    append_u32(bytes, 4);
    append_u32(bytes, static_cast<uint32_t>(metadata.size()));
    bytes.insert(bytes.end(), metadata.begin(), metadata.end());
    append_record(bytes, "post_patch_projection", 1, 1024, 0.1f);
    append_record(bytes, "post_block_0", 1, 1024, 0.2f);
    append_record(bytes, "post_vit", 1, 1024, 0.3f);
    append_record(bytes, "post_aligner", 1, 4096, 0.4f);
    return bytes;
}

std::vector<uint8_t> block_fixture() {
    const std::string metadata =
        "{\"block\": 0, \"image\": \"fixture.png\", "
        "\"lane\": \"bf16\", \"n_llm_h\": 1, \"n_llm_w\": 1, "
        "\"n_vit_h\": 1, \"n_vit_w\": 1, "
        "\"patch_digest\": \"0123456789abcdef0123456789abcdef\", "
        "\"source_revision\": \"47bede8\", "
        "\"weight_digest\": \"abcdef0123456789abcdef0123456789\"}";
    std::vector<uint8_t> bytes {'D','S','4','B','L','K',0,0};
    append_u32(bytes, 1);
    append_u32(bytes, 13);
    append_u32(bytes, static_cast<uint32_t>(metadata.size()));
    bytes.insert(bytes.end(), metadata.begin(), metadata.end());
    append_record(bytes, "norm1_out", 1, 1024, 0.01f);
    append_record(bytes, "qkv_biased", 1, 3072, 0.02f);
    append_record_shape(bytes, "q_roped", {1, 16, 64}, 0.03f);
    append_record_shape(bytes, "k_roped", {1, 16, 64}, 0.04f);
    append_record_shape(bytes, "v_in", {1, 16, 64}, 0.05f);
    append_record(bytes, "sdpa_out", 1, 1024, 0.06f);
    append_record(bytes, "wo_biased", 1, 1024, 0.07f);
    append_record(bytes, "post_attn_residual", 1, 1024, 0.08f);
    append_record(bytes, "norm2_out", 1, 1024, 0.09f);
    append_record(bytes, "mlp_gate", 1, 2816, 0.10f);
    append_record(bytes, "mlp_up", 1, 2816, 0.11f);
    append_record(bytes, "mlp_silu_act", 1, 2816, 0.12f);
    append_record(bytes, "mlp_down_out", 1, 1024, 0.13f);
    return bytes;
}

void test_parser_and_mutations() {
    auto bytes = fixture();
    Deepseek4VisionOracle oracle;
    std::string error;
    CHECK(deepseek4_parse_vision_oracle(bytes, oracle, error),
          "valid authority-lane oracle parses");
    CHECK(oracle.records.size() == 4 &&
              oracle.lane == Deepseek4VisionOracleLane::BF16 &&
              oracle.n_vit_h == 1 && oracle.n_llm_w == 1,
          "oracle metadata and record count survive parsing");
    const auto * aligner = deepseek4_vision_oracle_record(
        oracle, "post_aligner");
    CHECK(aligner && aligner->shape == std::vector<int64_t>({1, 4096}) &&
              aligner->values.size() == 4096,
          "post-aligner row-major shape is retained");
    const auto * patch = deepseek4_vision_oracle_record(
        oracle, "post_patch_projection");
    CHECK(patch && patch->checksum == 0x9fabca0173fb481eULL,
          "fixture checksum pins the standard FNV-1a 64 offset basis");

    auto corrupt = bytes;
    corrupt.back() ^= 1;
    CHECK(!deepseek4_parse_vision_oracle(corrupt, oracle, error),
          "payload checksum mutation is rejected");
    auto trailing = bytes;
    trailing.push_back(0);
    CHECK(!deepseek4_parse_vision_oracle(trailing, oracle, error),
          "trailing oracle bytes are rejected");
    auto invalid_json = bytes;
    const size_t metadata_colon = 20 +
        std::string(reinterpret_cast<const char *>(bytes.data() + 20),
                    bytes.size() - 20).find(':');
    invalid_json[metadata_colon] = '=';
    CHECK(!deepseek4_parse_vision_oracle(invalid_json, oracle, error),
          "metadata must be valid JSON rather than matching byte patterns");
}

void test_block_parser() {
    const auto bytes = block_fixture();
    Deepseek4VisionOracle oracle;
    std::string error;
    CHECK(deepseek4_parse_vision_block_oracle(bytes, oracle, error) &&
              oracle.block == 0 && oracle.records.size() == 13,
          "13-record block diagnostic oracle parses independently of v1");
    const auto * query = deepseek4_vision_oracle_record(oracle, "q_roped");
    CHECK(query && query->shape == std::vector<int64_t>({1, 16, 64}),
          "block oracle retains source row/head/dimension shape");
    CHECK(!deepseek4_parse_vision_oracle(bytes, oracle, error),
          "stable DS4ORC v1 parser rejects diagnostic DS4BLK magic");
}

void test_digest_and_comparison() {
    std::vector<uint16_t> empty;
    CHECK(deepseek4_vision_patch_digest(empty) ==
              "e3b0c44298fc1c149afbf4c8996fb924",
          "patch binding uses SHA-256 truncated to 128 bits");
    std::vector<uint16_t> values {0x3f80, 0xbf80};
    CHECK(deepseek4_vision_patch_digest(values) ==
              "150f7b8c669232bca67ec962d047fa4d",
          "patch digest hashes little-endian BF16 payload bytes");

    Deepseek4VisionOracleRecord record;
    record.values = {1.0f, 2.0f, 3.0f};
    std::vector<float> actual = {1.0f, 2.0f, 3.0f};
    Deepseek4VisionOracleComparison comparison;
    std::string error;
    CHECK(deepseek4_compare_vision_checkpoint(
              actual, record, comparison, error) &&
              comparison.relative_l2 == 0.0 && comparison.cosine == 1.0 &&
              comparison.max_abs == 0.0f,
          "identical checkpoint comparison is exact");
    actual[1] = 2.5f;
    CHECK(deepseek4_compare_vision_checkpoint(
              actual, record, comparison, error) &&
              comparison.relative_l2 > 0.0 &&
              comparison.max_abs == 0.5f &&
              comparison.max_abs_index == 1,
          "checkpoint comparison reports structural delta without a gate");

    Deepseek4VisionOracleComparison lane;
    lane.relative_l2 = 0.01;
    lane.cosine = 0.99;
    lane.max_abs = 0.1f;
    Deepseek4VisionOracleComparison engine;
    engine.relative_l2 = 0.0009;
    engine.cosine = 0.999;
    engine.max_abs = 0.01f;
    double ratio = 0.0;
    CHECK(deepseek4_vision_ratio_gate(engine, lane, 0.10, ratio) &&
              std::fabs(ratio - 0.09) < 1.0e-12,
          "pre-registered ratio gate accepts a value below its bound");
    engine.relative_l2 = 0.0011;
    CHECK(!deepseek4_vision_ratio_gate(engine, lane, 0.10, ratio) &&
              std::fabs(ratio - 0.11) < 1.0e-12,
          "ratio mutation crosses the bound and proves red is reachable");
    lane.relative_l2 = 0.0;
    CHECK(!deepseek4_vision_ratio_gate(engine, lane, 0.10, ratio) &&
              std::isinf(ratio),
          "zero authority-lane scale cannot vacuously pass the gate");

    engine.relative_l2 = 0.0009;
    double consumption = 0.0;
    CHECK(deepseek4_vision_budget_gate(
              engine, 0.001, consumption) &&
              std::fabs(consumption - 0.9) < 1.0e-12,
          "measured compounding budget accepts a residual below its bound");
    engine.relative_l2 = 0.0011;
    CHECK(!deepseek4_vision_budget_gate(
              engine, 0.001, consumption) &&
              std::fabs(consumption - 1.1) < 1.0e-12,
          "compounding-budget mutation crosses the bound");
    CHECK(!deepseek4_vision_budget_gate(engine, 0.0, consumption) &&
              std::isinf(consumption),
          "non-positive compounding budget cannot vacuously pass");
}

}  // namespace

int main() {
    test_parser_and_mutations();
    test_block_parser();
    test_digest_and_comparison();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
