#include "deepseek4_vision_markers.h"

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

using dflash::Deepseek4ImageMarkers;
using dflash::deepseek4_load_image_markers;

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
        char pattern[] = "/tmp/ember-ds4-markers-XXXXXX";
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
    MissingEnd,
    PadF16,
    PadTwoByTwo,
    PadExtraRow,
};

size_t write_marker_gguf(const std::filesystem::path & path,
                         Mutation mutation = Mutation::None) {
    ggml_init_params model_params {
        /*.mem_size=*/1024 * 1024,
        /*.mem_buffer=*/nullptr,
        /*.no_alloc=*/true,
    };
    ggml_context * model = ggml_init(model_params);
    gguf_context * gguf = gguf_init_empty();
    if (!model || !gguf) std::abort();

    constexpr const char * names[] = {
        "mm.image_begin.weight",
        "mm.image_pad.weight",
        "v.image_newline.weight",
        "mm.image_end.weight",
    };
    std::array<std::array<float, 8>, 4> data {};
    for (size_t marker = 0; marker < data.size(); ++marker) {
        for (size_t column = 0; column < data[marker].size(); ++column) {
            data[marker][column] = static_cast<float>(marker * 10 + column);
        }
    }
    for (size_t marker = 0; marker < data.size(); ++marker) {
        if (mutation == Mutation::MissingEnd && marker == 3) continue;
        ggml_tensor * tensor = nullptr;
        if (mutation == Mutation::PadF16 && marker == 1) {
            tensor = ggml_new_tensor_1d(model, GGML_TYPE_F16, 4);
        } else if (mutation == Mutation::PadTwoByTwo && marker == 1) {
            tensor = ggml_new_tensor_2d(model, GGML_TYPE_F32, 2, 2);
        } else if (mutation == Mutation::PadExtraRow && marker == 1) {
            tensor = ggml_new_tensor_2d(model, GGML_TYPE_F32, 4, 2);
        } else {
            tensor = ggml_new_tensor_1d(model, GGML_TYPE_F32, 4);
        }
        ggml_set_name(tensor, names[marker]);
        gguf_add_tensor(gguf, tensor);
        gguf_set_tensor_data(gguf, names[marker], data[marker].data());
    }
    if (!gguf_write_to_file(gguf, path.c_str(), false)) std::abort();
    const int64_t last = gguf_get_n_tensors(gguf) - 1;
    const size_t payload_end = gguf_get_meta_size(gguf) +
                               gguf_get_tensor_offset(gguf, last) +
                               gguf_get_tensor_size(gguf, last);
    gguf_free(gguf);
    ggml_free(model);
    return payload_end;
}

void test_exact_markers() {
    TempDir directory;
    const auto path = directory.path / "mmproj.gguf";
    write_marker_gguf(path);
    Deepseek4ImageMarkers markers;
    std::string error;
    CHECK(deepseek4_load_image_markers(path, 4, markers, error),
          "the four real mmproj marker names load");
    CHECK(markers.start == std::vector<float>({0, 1, 2, 3}),
          "image-begin values are copied exactly");
    CHECK(markers.pad == std::vector<float>({10, 11, 12, 13}),
          "image-pad values are copied exactly");
    CHECK(markers.newline == std::vector<float>({20, 21, 22, 23}),
          "image-newline values are copied exactly");
    CHECK(markers.end == std::vector<float>({30, 31, 32, 33}),
          "image-end values are copied exactly");
}

void test_contract_mutations_fail_closed() {
    const struct Case {
        Mutation mutation;
        const char * name;
    } cases[] = {
        {Mutation::MissingEnd, "a missing marker"},
        {Mutation::PadF16, "a non-F32 marker"},
        {Mutation::PadTwoByTwo, "a rank-2 square with the right element count"},
        {Mutation::PadExtraRow, "a marker with extra elements"},
    };
    for (const Case & test : cases) {
        TempDir directory;
        const auto path = directory.path / "mutated.gguf";
        write_marker_gguf(path, test.mutation);
        Deepseek4ImageMarkers markers;
        std::string error;
        CHECK(!deepseek4_load_image_markers(path, 4, markers, error),
              test.name);
        CHECK(markers.start.empty() && markers.pad.empty() &&
                  markers.newline.empty() && markers.end.empty(),
              "a rejected mmproj leaves no partial marker state");
    }
}

void test_truncation_and_special_file_fail_closed() {
    TempDir directory;
    const auto path = directory.path / "truncated.gguf";
    const size_t payload_end = write_marker_gguf(path);
    CHECK(payload_end > 0 &&
              truncate(path.c_str(), static_cast<off_t>(payload_end - 1)) == 0,
          "mmproj fixture is truncated inside the final marker payload");
    Deepseek4ImageMarkers markers;
    std::string error;
    CHECK(!deepseek4_load_image_markers(path, 4, markers, error),
          "a marker tensor extending past EOF is rejected");

    const auto fifo = directory.path / "mmproj.fifo";
    CHECK(mkfifo(fifo.c_str(), 0600) == 0, "mmproj FIFO fixture is created");
    CHECK(!deepseek4_load_image_markers(fifo, 4, markers, error) &&
              error == "DeepSeek4 mmproj GGUF is not a regular file",
          "a FIFO mmproj is rejected without waiting for a writer");
}

}  // namespace

int main() {
    test_exact_markers();
    test_contract_mutations_fail_closed();
    test_truncation_and_special_file_fail_closed();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
