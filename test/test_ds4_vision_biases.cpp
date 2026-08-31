#include "deepseek4_vision_biases.h"
#include "deepseek4_vision_contract.h"

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using dflash::Deepseek4VisionBiasInventory;
using dflash::deepseek4_inspect_vision_router_biases;

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
        char pattern[] = "/tmp/ember-ds4-vision-biases-XXXXXX";
        char * created = mkdtemp(pattern);
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

enum class Mutation {
    None,
    MissingLayer,
    WrongSuffix,
    WrongBlockSpelling,
    WrongType,
    WrongShape,
    OutOfRangeLayer,
    WrongArchitecture,
};

void add_bias(ggml_context * model, gguf_context * gguf,
              const std::string & name, ggml_type type, bool wrong_shape,
              const void * data) {
    ggml_tensor * tensor = wrong_shape
        ? ggml_new_tensor_2d(model, type, 4, 2)
        : ggml_new_tensor_1d(model, type, 4);
    ggml_set_name(tensor, name.c_str());
    gguf_add_tensor(gguf, tensor);
    gguf_set_tensor_data(gguf, name.c_str(), data);
}

void write_bias_gguf(const std::filesystem::path & path,
                     Mutation mutation = Mutation::None) {
    ggml_init_params params {
        /*.mem_size=*/1024 * 1024,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/true,
    };
    ggml_context * model = ggml_init(params);
    gguf_context * gguf = gguf_init_empty();
    if (!model || !gguf) std::abort();
    gguf_set_val_str(gguf, "general.architecture",
                     mutation == Mutation::WrongArchitecture
                         ? "qwen4exp" : "deepseek4");
    std::array<float, 8> data {};
    for (int layer = 0; layer < 3; ++layer) {
        if (mutation == Mutation::MissingLayer && layer == 1) continue;
        const std::string suffix =
            mutation == Mutation::WrongSuffix && layer == 1
                ? "exp_probs_b_vl"
                : dflash::DEEPSEEK4_VISION_ROUTER_BIAS_SUFFIX;
        add_bias(model, gguf,
                 "blk." + std::to_string(layer) + "." + suffix,
                 mutation == Mutation::WrongType && layer == 1
                     ? GGML_TYPE_F16 : GGML_TYPE_F32,
                 mutation == Mutation::WrongShape && layer == 1,
                 data.data());
    }
    if (mutation == Mutation::WrongBlockSpelling) {
        add_bias(model, gguf,
                 std::string("blk.01.") +
                     dflash::DEEPSEEK4_VISION_ROUTER_BIAS_SUFFIX,
                 GGML_TYPE_F32, false, data.data());
    }
    if (mutation == Mutation::OutOfRangeLayer) {
        add_bias(model, gguf,
                 std::string("blk.3.") +
                     dflash::DEEPSEEK4_VISION_ROUTER_BIAS_SUFFIX,
                 GGML_TYPE_F32, false, data.data());
    }
    if (!gguf_write_to_file(gguf, path.c_str(), false)) std::abort();
    gguf_free(gguf);
    ggml_free(model);
}

void test_exact_inventory() {
    TempDir directory;
    const auto path = directory.path / "model.gguf";
    write_bias_gguf(path);
    Deepseek4VisionBiasInventory inventory;
    std::string error;
    CHECK(deepseek4_inspect_vision_router_biases(
              path, 3, 4, inventory, error),
          "exact converted vision-bias inventory validates");
    CHECK(inventory.layer_ids == std::vector<int>({0, 1, 2}),
          "vision-bias inventory covers every block exactly once");
}

void test_mutations_fail_closed() {
    constexpr Mutation mutations[] = {
        Mutation::MissingLayer,
        Mutation::WrongSuffix,
        Mutation::WrongBlockSpelling,
        Mutation::WrongType,
        Mutation::WrongShape,
        Mutation::OutOfRangeLayer,
        Mutation::WrongArchitecture,
    };
    for (Mutation mutation : mutations) {
        TempDir directory;
        const auto path = directory.path / "mutated.gguf";
        write_bias_gguf(path, mutation);
        Deepseek4VisionBiasInventory inventory;
        std::string error;
        CHECK(!deepseek4_inspect_vision_router_biases(
                  path, 3, 4, inventory, error),
              "mutated vision-bias inventory is rejected");
        CHECK(inventory.layer_ids.empty(),
              "a rejected inventory leaves no partial layer state");
    }
}

void test_special_files_fail_closed() {
    TempDir directory;
    const auto fifo = directory.path / "model.fifo";
    CHECK(mkfifo(fifo.c_str(), 0600) == 0,
          "vision-bias FIFO fixture is created");
    Deepseek4VisionBiasInventory inventory;
    std::string error;
    CHECK(!deepseek4_inspect_vision_router_biases(
              fifo, 3, 4, inventory, error) &&
              error == "DeepSeek4 language GGUF is not a regular file",
          "a FIFO is rejected without waiting for a writer");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2) {
        Deepseek4VisionBiasInventory inventory;
        std::string error;
        if (!deepseek4_inspect_vision_router_biases(
                argv[1], 43, 256, inventory, error)) {
            std::fprintf(stderr, "real vision-bias inventory failed: %s\n",
                         error.c_str());
            return 1;
        }
        std::printf("vision_router_biases %zu blocks 0..42 F32[256]\n",
                    inventory.layer_ids.size());
        return 0;
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s [real-language.gguf]\n", argv[0]);
        return 2;
    }
    test_exact_inventory();
    test_mutations_fail_closed();
    test_special_files_fail_closed();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
