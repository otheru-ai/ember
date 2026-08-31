// CPU execution proof for the layer-major vision MoE row partition.
//
// The production graph gathers text and learned-image rows, evaluates their
// different routing contracts, scatters both subsets back to the original
// prompt rows, then adds one shared-expert result. This fixture exercises the
// same ggml shape/lifetime pattern with distinct scalar transforms so a row
// swap, missing zero-fill, or stale row-index input cannot pass silently.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (condition) {                                                    \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__,     \
                         (message));                                        \
        }                                                                   \
    } while (0)

int main() {
    constexpr int kWidth = 3;
    constexpr int kRows = 5;
    const std::vector<float> input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f,
    };
    std::vector<int32_t> text_rows = {0, 2, 4};
    std::vector<int32_t> vision_rows = {1, 3};

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "CPU backend initializes");
    if (!backend) return 1;

    ggml_init_params params{};
    params.mem_size = 4U * 1024U * 1024U;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "ggml context initializes");
    if (!ctx) {
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor * all = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, kWidth, kRows);
    ggml_tensor * text_idx = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, static_cast<int64_t>(text_rows.size()));
    ggml_tensor * vision_idx = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, static_cast<int64_t>(vision_rows.size()));
    ggml_set_input(all);
    ggml_set_input(text_idx);
    ggml_set_input(vision_idx);

    ggml_tensor * text = ggml_get_rows(ctx, all, text_idx);
    ggml_tensor * vision = ggml_get_rows(ctx, all, vision_idx);
    text = ggml_scale(ctx, text, 100.0f);
    vision = ggml_scale(ctx, vision, 10.0f);
    ggml_tensor * text_scatter = ggml_get_rows_back(
        ctx, ggml_cont(ctx, text), text_idx, all);
    ggml_tensor * vision_scatter = ggml_get_rows_back(
        ctx, ggml_cont(ctx, vision), vision_idx, all);
    ggml_tensor * routed = ggml_add(ctx, text_scatter, vision_scatter);
    ggml_tensor * output = ggml_add(ctx, routed, all);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "CPU graph tensors allocate");
    if (buffer) {
        ggml_backend_tensor_set(all, input.data(), 0,
                                input.size() * sizeof(float));
        ggml_backend_tensor_set(text_idx, text_rows.data(), 0,
                                text_rows.size() * sizeof(int32_t));
        ggml_backend_tensor_set(vision_idx, vision_rows.data(), 0,
                                vision_rows.size() * sizeof(int32_t));

        // The graph owns backend copies. Destroying the host vectors before
        // compute makes a deferred or borrowed binding fail observably.
        text_rows.clear();
        text_rows.shrink_to_fit();
        vision_rows.clear();
        vision_rows.shrink_to_fit();

        const bool computed =
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        CHECK(computed, "mixed-row graph computes after host inputs die");
        if (computed) {
            std::vector<float> actual(input.size());
            ggml_backend_tensor_get(output, actual.data(), 0,
                                    actual.size() * sizeof(float));
            bool exact = true;
            for (int row = 0; row < kRows; ++row) {
                const float factor = (row == 1 || row == 3) ? 11.0f : 101.0f;
                for (int col = 0; col < kWidth; ++col) {
                    const size_t i = static_cast<size_t>(row * kWidth + col);
                    exact = exact && actual[i] == input[i] * factor;
                }
            }
            CHECK(exact,
                  "gathered routes scatter to original rows and add shared once");
        }
        ggml_backend_buffer_free(buffer);
    }

    ggml_free(ctx);
    ggml_backend_free(backend);
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
