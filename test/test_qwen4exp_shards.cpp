#include "qwen4exp_model.h"

#include "ggml.h"
#include "gguf.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using dflash::common::qwen4exp_discover_gguf_shards;
using dflash::common::validate_qwen4exp_gguf;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message) do {                                      \
    if (condition) { ++g_pass; } else {                                     \
        ++g_fail; std::fprintf(stderr, "FAIL: %s\n", message);             \
    }                                                                        \
} while (0)

namespace {

struct TempDir {
    std::filesystem::path path;

    TempDir() {
        char pattern[] = "/tmp/ember-qwen-shards-XXXXXX";
        char * created = ::mkdtemp(pattern);
        if (!created) {
            std::fprintf(stderr, "mkdtemp failed: %s\n", std::strerror(errno));
            std::abort();
        }
        path = created;
    }

    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

void write_byte(const std::filesystem::path & path) {
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file || std::fputc(0, file) == EOF || std::fclose(file) != 0) {
        std::fprintf(stderr, "failed to create %s\n", path.c_str());
        std::abort();
    }
}

void write_tiny_gguf(const std::filesystem::path & path, uint16_t split_no,
                     uint16_t split_count, int32_t tensor_count,
                     const char * tensor_name, const char * invariant,
                     bool include_tensor_data = true,
                     bool full_metadata = true,
                     bool stock_metadata = false,
                     bool injected_metadata = false,
                     gguf_type canonical_array_type = GGUF_TYPE_COUNT,
                     int canonical_array_mutation = 0) {
    ggml_init_params model_params{
        /*.mem_size=*/1024 * 1024,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/true,
    };
    ggml_context * model = ggml_init(model_params);
    gguf_context * gguf = gguf_init_empty();
    if (!model || !gguf) std::abort();
    if (full_metadata) {
        gguf_set_val_str(gguf, "general.architecture", "qwen4exp");
        gguf_set_val_str(gguf, "test.invariant", invariant);
    }
    if (stock_metadata) {
        gguf_set_val_u32(gguf, "general.quantization_version", 2);
        gguf_set_val_u32(gguf, "general.file_type", 118);
        gguf_set_val_str(gguf, "ember.intervention.kind", "none_control");
        gguf_set_val_str(
            gguf, "ember.intervention.release_eligibility",
            "control_only_requires_manifest_for_release");
    }
    if (injected_metadata) {
        gguf_set_val_str(gguf, "test.injected", "not-authoritative");
    }
    if (canonical_array_type != GGUF_TYPE_COUNT) {
        const struct RequiredU32 { const char * key; uint32_t value; } values[] = {
            {"qwen4exp.context_length", 262144},
            {"qwen4exp.embedding_length", 2560},
            {"qwen4exp.block_count", 48},
            {"qwen4exp.attention.head_count", 24},
            {"qwen4exp.attention.head_count_kv", 2},
            {"qwen4exp.attention.key_length", 256},
            {"qwen4exp.rope.dimension_count", 64},
            {"qwen4exp.expert_count", 512},
            {"qwen4exp.expert_used_count", 10},
            {"qwen4exp.expert_feed_forward_length", 640},
            {"qwen4exp.expert_shared_feed_forward_length", 640},
            {"qwen4exp.ssm.conv_kernel", 4},
            {"qwen4exp.ssm.inner_size", 6144},
            {"qwen4exp.ssm.state_size", 128},
            {"qwen4exp.ssm.time_step_rank", 48},
            {"qwen4exp.ssm.group_count", 16},
            {"qwen4exp.hyper_connection.count", 4},
            {"qwen4exp.hyper_connection.low_rank", 320},
            {"qwen4exp.attention.indexer.head_count", 4},
            {"qwen4exp.attention.indexer.key_length", 128},
            {"qwen4exp.attention.indexer.top_k", 2048},
            {"qwen4exp.ple.ngram_size", 3},
            {"qwen4exp.ple.heads_per_ngram", 8},
            {"qwen4exp.ple.conv_kernel", 4},
            {"qwen4exp.ple.eos_token_id", 248044},
            {"qwen4exp.ple.image_token_id", 248056},
            {"qwen4exp.embedding_length_per_layer_input", 160},
        };
        for (const auto & value : values) {
            gguf_set_val_u32(gguf, value.key, value.value);
        }
        gguf_set_val_f32(gguf, "qwen4exp.rope.freq_base", 10000000.0f);
        gguf_set_val_f32(
            gguf, "qwen4exp.attention.layer_norm_rms_epsilon", 1.0e-6f);

        int32_t ratios_i32[48] = {};
        uint32_t ratios_u32[48] = {};
        for (size_t i = 3; i < 48; i += 4) {
            ratios_i32[i] = 4;
            ratios_u32[i] = 4;
        }
        if (canonical_array_mutation == 1) ratios_i32[3] = -4;
        if (canonical_array_mutation == 2) {
            ratios_i32[3] = 3;
            ratios_u32[3] = 3;
        }
        const int32_t rope_sections[] = {11, 11, 10};
        const int32_t ple_layers[] = {1};
        const float ratios_f32[48] = {};
        const void * ratio_data = ratios_i32;
        if (canonical_array_type == GGUF_TYPE_UINT32) ratio_data = ratios_u32;
        if (canonical_array_type == GGUF_TYPE_FLOAT32) ratio_data = ratios_f32;
        const size_t ratio_count = canonical_array_mutation == 3 ? 47 : 48;
        gguf_set_arr_data(gguf, "qwen4exp.attention.compress_ratios",
                          canonical_array_type, ratio_data, ratio_count);
        gguf_set_arr_data(gguf, "qwen4exp.rope.dimension_sections",
                          GGUF_TYPE_INT32, rope_sections, 3);
        gguf_set_arr_data(gguf, "qwen4exp.ple.layers", GGUF_TYPE_INT32,
                          ple_layers, 1);

        const uint64_t vocab_sizes[] = {
            20000003, 20000023, 20000033, 20000047,
            20000059, 20000063, 20000069, 20000077,
            20000081, 20000093, 20000107, 20000147,
            20000153, 20000159, 20000161, 20000171,
        };
        uint64_t offsets[16] = {};
        for (size_t i = 1; i < 16; ++i) {
            offsets[i] = offsets[i - 1] + vocab_sizes[i - 1];
        }
        const uint64_t multipliers[] = {
            23703573157769ULL, 20109073645365ULL, 8052911324071ULL,
        };
        gguf_set_arr_data(gguf, "qwen4exp.ple.head_offsets",
                          GGUF_TYPE_UINT64, offsets, 16);
        gguf_set_arr_data(gguf, "qwen4exp.ple.head_vocab_sizes",
                          GGUF_TYPE_UINT64, vocab_sizes, 16);
        gguf_set_arr_data(gguf, "qwen4exp.ple.layer_multipliers",
                          GGUF_TYPE_UINT64, multipliers, 3);
    }
    gguf_set_val_u16(gguf, "split.no", split_no);
    gguf_set_val_u16(gguf, "split.count", split_count);
    gguf_set_val_i32(gguf, "split.tensors.count", tensor_count);
    ggml_tensor * tensor = ggml_new_tensor_1d(model, GGML_TYPE_F32, 1);
    ggml_set_name(tensor, tensor_name);
    gguf_add_tensor(gguf, tensor);
    if (!gguf_write_to_file(gguf, path.c_str(), true)) std::abort();

    if (include_tensor_data) {
        // A writer context has no file-derived data_offset yet. Its serialized
        // metadata size is the actual aligned tensor-data origin.
        const size_t end = gguf_get_meta_size(gguf) +
                           gguf_get_tensor_offset(gguf, 0) +
                           gguf_get_tensor_size(gguf, 0);
        const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
        if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(end)) != 0) std::abort();
        ::close(fd);
    }
    gguf_free(gguf);
    ggml_free(model);
}

std::filesystem::path shard(const TempDir & dir, int number, int count) {
    char name[80];
    std::snprintf(name, sizeof(name), "model-%05d-of-%05d.gguf", number, count);
    return dir.path / name;
}

} // namespace

int main() {
    {
        TempDir dir;
        const auto model = dir.path / "model.gguf";
        write_tiny_gguf(model, 0, 1, 1, "one.weight", "same", true,
                        true, false, false, GGUF_TYPE_INT32);
        std::string error;
        CHECK(!validate_qwen4exp_gguf(model.c_str(), error) &&
                  error.find("tokenizer vocabulary") != std::string::npos,
              "canonical llama.cpp INT32 architecture arrays are accepted");
    }
    {
        TempDir dir;
        const auto model = dir.path / "model.gguf";
        write_tiny_gguf(model, 0, 1, 1, "one.weight", "same", true,
                        true, false, false, GGUF_TYPE_UINT32);
        std::string error;
        CHECK(!validate_qwen4exp_gguf(model.c_str(), error) &&
                  error.find("tokenizer vocabulary") != std::string::npos,
              "legacy UINT32 architecture arrays remain accepted");
    }
    {
        TempDir dir;
        const auto model = dir.path / "model.gguf";
        write_tiny_gguf(model, 0, 1, 1, "one.weight", "same", true,
                        true, false, false, GGUF_TYPE_INT32, 1);
        std::string error;
        CHECK(!validate_qwen4exp_gguf(model.c_str(), error) &&
                  error.find("compress_ratios[3]") != std::string::npos,
              "negative signed architecture-array values fail closed");
    }
    {
        TempDir dir;
        const auto model = dir.path / "model.gguf";
        write_tiny_gguf(model, 0, 1, 1, "one.weight", "same", true,
                        true, false, false, GGUF_TYPE_UINT32, 2);
        std::string error;
        CHECK(!validate_qwen4exp_gguf(model.c_str(), error) &&
                  error.find("compress_ratios[3]") != std::string::npos,
              "wrong unsigned architecture-array values fail closed");
    }
    {
        TempDir dir;
        const auto model = dir.path / "model.gguf";
        write_tiny_gguf(model, 0, 1, 1, "one.weight", "same", true,
                        true, false, false, GGUF_TYPE_INT32, 3);
        std::string error;
        CHECK(!validate_qwen4exp_gguf(model.c_str(), error) &&
                  error.find("incompatible Qwen4Exp array") !=
                      std::string::npos,
              "short architecture arrays fail closed");
    }
    {
        TempDir dir;
        const auto model = dir.path / "model.gguf";
        write_tiny_gguf(model, 0, 1, 1, "one.weight", "same", true,
                        true, false, false, GGUF_TYPE_FLOAT32);
        std::string error;
        CHECK(!validate_qwen4exp_gguf(model.c_str(), error) &&
                  error.find("incompatible Qwen4Exp array") !=
                      std::string::npos,
              "wrong architecture-array element types fail closed");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 3);
        const auto second = shard(dir, 2, 3);
        const auto third = shard(dir, 3, 3);
        write_byte(first); write_byte(second); write_byte(third);
        std::vector<std::string> paths;
        std::string error;
        CHECK(qwen4exp_discover_gguf_shards(second.c_str(), paths, error),
              "any canonical shard discovers its complete set");
        CHECK(paths.size() == 3 && paths[0] == first.string() &&
                  paths[1] == second.string() && paths[2] == third.string(),
              "canonical shards are returned in exact numeric order");
    }
    {
        TempDir dir;
        write_byte(shard(dir, 1, 2));
        std::vector<std::string> paths;
        std::string error;
        CHECK(!qwen4exp_discover_gguf_shards(shard(dir, 1, 2).c_str(), paths,
                                             error) &&
                  error.find("missing") != std::string::npos,
              "a missing canonical shard fails closed");
    }
    {
        TempDir dir;
        write_byte(shard(dir, 1, 2)); write_byte(shard(dir, 2, 2));
        write_byte(shard(dir, 1, 3));
        std::vector<std::string> paths;
        std::string error;
        CHECK(!qwen4exp_discover_gguf_shards(shard(dir, 2, 2).c_str(), paths,
                                             error) &&
                  error.find("mismatched filename shard count") !=
                      std::string::npos,
              "one stem cannot advertise conflicting shard counts");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_byte(first);
        if (::link(first.c_str(), second.c_str()) != 0) std::abort();
        std::vector<std::string> paths;
        std::string error;
        CHECK(!qwen4exp_discover_gguf_shards(first.c_str(), paths, error) &&
                  error.find("duplicate inode") != std::string::npos,
              "hard-linked shard aliases fail closed");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "one.weight", "same");
        write_tiny_gguf(second, 0, 2, 2, "two.weight", "same");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(second.c_str(), error) &&
                  error.find("split.no/count") != std::string::npos,
              "split.no is checked against filename order");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 3, "one.weight", "same");
        write_tiny_gguf(second, 1, 2, 3, "two.weight", "same");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("global tensor inventory") != std::string::npos,
              "declared aggregate tensor inventory is exact");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "same.weight", "same");
        write_tiny_gguf(second, 1, 2, 2, "same.weight", "same");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("duplicate or invalid tensor") != std::string::npos,
              "duplicate tensor names across shards fail closed");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "one.weight", "left");
        write_tiny_gguf(second, 1, 2, 2, "two.weight", "right");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("test.invariant") != std::string::npos,
              "repeated invariant metadata must match exactly across shards");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 8);
        write_tiny_gguf(first, 0, 8, 8, "one.weight", "authoritative",
                        true, true, true);
        for (int number = 2; number <= 8; ++number) {
            const std::string tensor = "tensor." + std::to_string(number);
            write_tiny_gguf(shard(dir, number, 8),
                            static_cast<uint16_t>(number - 1), 8, 8,
                            tensor.c_str(), "unused", true, false, true);
        }
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("continuation metadata differs") ==
                      std::string::npos &&
                  error.find("qwen4exp.context_length") != std::string::npos,
              "canonical eight-shard stock metadata reaches model validation");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "one.weight", "authoritative",
                        true, true, true);
        write_tiny_gguf(second, 1, 2, 2, "two.weight", "unused", true,
                        false, true, true);
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("test.injected") != std::string::npos,
              "continuation metadata absent from shard one fails closed");
    }
    {
        TempDir dir;
        const auto first = shard(dir, 1, 2);
        const auto second = shard(dir, 2, 2);
        write_tiny_gguf(first, 0, 2, 2, "one.weight", "same", false);
        write_tiny_gguf(second, 1, 2, 2, "two.weight", "same");
        std::string error;
        CHECK(!validate_qwen4exp_gguf(first.c_str(), error) &&
                  error.find("extends past end") != std::string::npos,
              "every shard tensor payload is checked against its own file");
    }

    std::printf("qwen4exp shards: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
