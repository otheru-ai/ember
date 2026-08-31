// Fixed-input gfx1151 oracle for ROCMI4's three device-side decoders.
//
// The full-model F32 diagnostic exposed a large disagreement but changed every
// quantized matmul at once.  This test isolates the primitive instead: the
// convert.cu dequantizer, production MMVQ, and production MMQ all consume the
// same hand-packed ROCMI4 blocks and are checked against the CPU decoder plus
// scalar F32 matmul.  Both dense MUL_MAT and routed MUL_MAT_ID are covered.
//
// The activations are exact Q8_1 values: every K32 block contains +/-127 times
// 0.25, and all other values are integral multiples of 0.25.  The device
// quantizers therefore choose the exactly representable scale 0.25 and incur
// zero activation-quantization error.  ROCMI4 scales are powers of two and all
// dot products remain in F32's exact-integer range.  The predeclared numeric
// budget is consequently zero, not a tolerance chosen after seeing hardware.
// The exact cases include the shipped K dimensions whose 32-value block counts
// leave partial MMQ_ITER_K=256 iterations, plus a full-iteration control.  A
// second case uses non-grid activations and compares MMVQ with MMQ directly.
// Its per-output budget is one full Q8_1 quantization step at every activation,
// weighted by that output row's L1 weight norm.  This is fixed from the input
// and CPU-decoded weights before either device result is inspected.  The bound
// requires both scales to span QK8_1 values; gfx1151 ROCMI4 MMQ uses D4, whose
// vals_per_scale is 32.  A D2S6 layout spans 64 and must not reuse this bound.

#include "convert.cuh"
#include "ggml-backend.h"
#include "ggml.h"
#include "rocmfpx.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message) do {                                      \
    if (condition) {                                                        \
        ++g_pass;                                                           \
    } else {                                                               \
        ++g_fail;                                                           \
        std::fprintf(stderr, "FAIL: %s\n", message);                      \
    }                                                                      \
} while (0)

namespace {

constexpr int64_t kFullIterationK = 256;
constexpr int64_t kRows = 128;
constexpr int64_t kExperts = 3;
constexpr int64_t kMmvqWidth = 1;
constexpr int64_t kMmqWidth = 16;
constexpr float kActivationScale = 0.25f;
constexpr float kMaxAbsBudget = 0.0f;
constexpr int64_t kExactKDims[] = {kFullIterationK, 160, 320, 640};

template <size_t N>
constexpr bool exact_fixture_k_dims_are_safe(const int64_t (&dims)[N]) {
    // Products are multiples of 2^-4. In those units the largest term is
    // 127 * 0.25 * 8 * 1.0 / 0.0625 = 4064. Staying below 2^24 makes every
    // possible accumulation order exact in F32, which is the zero-error proof.
    constexpr int64_t max_product_units = 127 * 8 * 4;
    for (size_t i = 0; i < N; ++i) {
        if (dims[i] % QK_ROCMI4 != 0 ||
            dims[i] * max_product_units >= (INT64_C(1) << 24)) {
            return false;
        }
    }
    return true;
}

static_assert(exact_fixture_k_dims_are_safe(kExactKDims),
              "exact ROCMI4 oracle K exceeds its zero-error F32 proof");

struct Fixture {
    int64_t k = 0;
    std::vector<block_rocmi4> packed;
    std::vector<float> decoded;
};

static int signed_nibble(int64_t expert, int64_t row, int64_t block,
                         int64_t index, int salt) {
    const int value = static_cast<int>(
        (expert * 11 + row * 7 + block * 5 + index * salt) & 15);
    return value < 8 ? value : value - 16;
}

static Fixture make_weights(int64_t experts, int64_t k) {
    const int64_t blocks_per_row = k / QK_ROCMI4;
    Fixture fixture;
    fixture.k = k;
    fixture.packed.resize(static_cast<size_t>(
        experts * kRows * blocks_per_row));
    fixture.decoded.resize(static_cast<size_t>(experts * kRows * k));

    constexpr uint8_t scales[] = {0x30, 0x38, 0x40}; // 0.25, 0.5, 1.0
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < kRows; ++row) {
            block_rocmi4 * packed_row = fixture.packed.data() +
                (expert * kRows + row) * blocks_per_row;
            float * decoded_row = fixture.decoded.data() +
                (expert * kRows + row) * k;
            for (int64_t block = 0; block < blocks_per_row; ++block) {
                block_rocmi4 & value = packed_row[block];
                value.e = scales[static_cast<size_t>(
                    (expert + row + block) % 3)];
                for (int64_t index = 0; index < QK_ROCMI4 / 2; ++index) {
                    const int lo = signed_nibble(
                        expert, row, block, index, 3);
                    const int hi = signed_nibble(
                        expert, row, block, index, 5);
                    value.qs[index] = static_cast<uint8_t>(
                        (lo & 15) | ((hi & 15) << 4));
                }
            }
            rocmfpx_dequantize_row_i4(
                packed_row, decoded_row, k);
        }
    }
    return fixture;
}

static std::vector<float> make_activations(int64_t k_dim, int64_t width) {
    std::vector<float> values(static_cast<size_t>(k_dim * width));
    for (int64_t token = 0; token < width; ++token) {
        for (int64_t k = 0; k < k_dim; ++k) {
            const int64_t within = k % QK8_1;
            int q;
            if (within == 0) {
                q = 127;
            } else if (within == 1) {
                q = -127;
            } else {
                q = static_cast<int>((token * 29 + k * 17) % 253) - 126;
            }
            values[static_cast<size_t>(token * k_dim + k)] =
                static_cast<float>(q) * kActivationScale;
        }
    }
    return values;
}

static std::vector<float> make_inexact_activation_row(int64_t k_dim) {
    std::vector<float> values(static_cast<size_t>(k_dim));
    for (int64_t k = 0; k < k_dim; ++k) {
        const float fk = static_cast<float>(k);
        values[static_cast<size_t>(k)] =
            0.73f * std::sin(0.071f * fk + 0.19f) +
            0.19f * std::cos(0.037f * fk + 0.11f) +
            0.003f * static_cast<float>((k % 11) - 5);
    }
    return values;
}

static std::vector<float> repeat_activation(
        const std::vector<float> & row, int64_t width) {
    const int64_t k_dim = static_cast<int64_t>(row.size());
    std::vector<float> values(static_cast<size_t>(k_dim * width));
    for (int64_t token = 0; token < width; ++token) {
        std::copy(row.begin(), row.end(),
                  values.begin() + static_cast<std::ptrdiff_t>(token * k_dim));
    }
    return values;
}

static std::vector<float> dense_reference(const Fixture & weights,
                                          const std::vector<float> & input,
                                          int64_t width) {
    const int64_t k_dim = weights.k;
    std::vector<float> output(static_cast<size_t>(kRows * width), 0.0f);
    for (int64_t token = 0; token < width; ++token) {
        for (int64_t row = 0; row < kRows; ++row) {
            float sum = 0.0f;
            for (int64_t k = 0; k < k_dim; ++k) {
                sum += weights.decoded[static_cast<size_t>(row * k_dim + k)] *
                       input[static_cast<size_t>(token * k_dim + k)];
            }
            output[static_cast<size_t>(token * kRows + row)] = sum;
        }
    }
    return output;
}

static std::vector<float> routed_reference(const Fixture & weights,
                                           const std::vector<float> & input,
                                           const std::vector<int32_t> & ids,
                                           int64_t width) {
    const int64_t k_dim = weights.k;
    std::vector<float> output(static_cast<size_t>(kRows * width), 0.0f);
    for (int64_t token = 0; token < width; ++token) {
        const int64_t expert = ids[static_cast<size_t>(token)];
        for (int64_t row = 0; row < kRows; ++row) {
            float sum = 0.0f;
            const size_t weight_base = static_cast<size_t>(
                (expert * kRows + row) * k_dim);
            for (int64_t k = 0; k < k_dim; ++k) {
                sum += weights.decoded[weight_base + static_cast<size_t>(k)] *
                       input[static_cast<size_t>(token * k_dim + k)];
            }
            output[static_cast<size_t>(token * kRows + row)] = sum;
        }
    }
    return output;
}

static bool exact_vectors(const std::vector<float> & actual,
                          const std::vector<float> & expected,
                          const char * label) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: size mismatch\n", label);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const float delta = std::abs(actual[i] - expected[i]);
        if (!(delta <= kMaxAbsBudget)) {
            std::fprintf(stderr,
                         "%s: first mismatch index=%zu actual=%g expected=%g "
                         "delta=%g budget=%g\n",
                         label, i, actual[i], expected[i], delta,
                         kMaxAbsBudget);
            return false;
        }
    }
    return true;
}

static ggml_context * make_context() {
    ggml_init_params params{};
    params.mem_size = 4U * 1024U * 1024U;
    params.no_alloc = true;
    return ggml_init(params);
}

static bool run_device_dequant(ggml_backend_t backend,
                               const Fixture & fixture) {
    ggml_context * ctx = make_context();
    if (!ctx) return false;
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_Q4_0_ROCMI4, fixture.k, kRows);
    ggml_tensor * decoded = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, fixture.k, kRows);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(weights, fixture.packed.data(), 0,
                            fixture.packed.size() * sizeof(block_rocmi4));
    ggml_backend_synchronize(backend);

    const to_fp32_cuda_t device_decode =
        ggml_get_to_fp32_cuda(GGML_TYPE_Q4_0_ROCMI4);
    bool ok = device_decode != nullptr;
    if (ok) {
        device_decode(weights->data, static_cast<float *>(decoded->data),
                      ggml_nelements(weights), nullptr);
        ok = hipDeviceSynchronize() == hipSuccess;
    }
    std::vector<float> actual(fixture.decoded.size());
    if (ok) {
        ggml_backend_tensor_get(decoded, actual.data(), 0,
                                actual.size() * sizeof(float));
        char label[64];
        std::snprintf(label, sizeof(label), "convert.cu ROCMI4 K=%lld",
                      static_cast<long long>(fixture.k));
        ok = exact_vectors(actual, fixture.decoded, label);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool run_dense(ggml_backend_t backend, const Fixture & fixture,
                      int64_t width,
                      const std::vector<float> * supplied_activation = nullptr,
                      std::vector<float> * captured = nullptr) {
    ggml_context * ctx = make_context();
    if (!ctx) return false;
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_Q4_0_ROCMI4, fixture.k, kRows);
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, fixture.k, width);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(weights, "rocmi4_oracle_dense_weight");
    ggml_set_name(output, "rocmi4_oracle_dense_output");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> activation = supplied_activation
        ? *supplied_activation : make_activations(fixture.k, width);
    const std::vector<float> expected = supplied_activation
        ? std::vector<float>() : dense_reference(fixture, activation, width);
    ggml_backend_tensor_set(weights, fixture.packed.data(), 0,
                            fixture.packed.size() * sizeof(block_rocmi4));
    ggml_backend_tensor_set(input, activation.data(), 0,
                            activation.size() * sizeof(float));
    const bool supported = ggml_backend_supports_op(backend, output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = supported
        ? ggml_backend_graph_compute(backend, graph)
        : GGML_STATUS_FAILED;
    std::vector<float> actual(static_cast<size_t>(kRows * width));
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(float));
        if (supplied_activation) {
            if (captured) *captured = actual;
        } else {
            char label[64];
            std::snprintf(label, sizeof(label), "dense %s K=%lld",
                          width == kMmvqWidth ? "MMVQ" : "MMQ",
                          static_cast<long long>(fixture.k));
            ok = exact_vectors(
                actual, expected, label);
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool run_routed(ggml_backend_t backend, const Fixture & fixture,
                       int64_t width,
                       const std::vector<float> * supplied_activation = nullptr,
                       const std::vector<int32_t> * supplied_ids = nullptr,
                       std::vector<float> * captured = nullptr) {
    ggml_context * ctx = make_context();
    if (!ctx) return false;
    ggml_tensor * weights = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q4_0_ROCMI4, fixture.k, kRows, kExperts);
    ggml_tensor * input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, fixture.k, 1, width);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, 1, width);
    ggml_tensor * output = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_name(weights, "rocmi4_oracle_expert_weight");
    ggml_set_name(output, "rocmi4_oracle_expert_output");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> activation = supplied_activation
        ? *supplied_activation : make_activations(fixture.k, width);
    std::vector<int32_t> expert_ids;
    if (supplied_ids) {
        expert_ids = *supplied_ids;
    } else {
        expert_ids.resize(static_cast<size_t>(width));
        for (int64_t token = 0; token < width; ++token) {
            expert_ids[static_cast<size_t>(token)] =
                static_cast<int32_t>(token % kExperts);
        }
    }
    const std::vector<float> expected = supplied_activation
        ? std::vector<float>()
        : routed_reference(fixture, activation, expert_ids, width);
    ggml_backend_tensor_set(weights, fixture.packed.data(), 0,
                            fixture.packed.size() * sizeof(block_rocmi4));
    ggml_backend_tensor_set(input, activation.data(), 0,
                            activation.size() * sizeof(float));
    ggml_backend_tensor_set(ids, expert_ids.data(), 0,
                            expert_ids.size() * sizeof(int32_t));
    const bool supported = ggml_backend_supports_op(backend, output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = supported
        ? ggml_backend_graph_compute(backend, graph)
        : GGML_STATUS_FAILED;
    std::vector<float> actual(static_cast<size_t>(kRows * width));
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(float));
        if (supplied_activation) {
            if (captured) *captured = actual;
        } else {
            char label[64];
            std::snprintf(label, sizeof(label), "routed %s K=%lld",
                          width == kMmvqWidth ? "MMVQ" : "MMQ",
                          static_cast<long long>(fixture.k));
            ok = exact_vectors(
                actual, expected, label);
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static std::vector<float> one_step_output_budget(
        const Fixture & weights, const std::vector<float> & activation,
        int64_t expert) {
    std::vector<float> budget(static_cast<size_t>(kRows), 0.0f);
    for (int64_t block = 0; block < weights.k / QK8_1; ++block) {
        float amax = 0.0f;
        for (int64_t k = 0; k < QK8_1; ++k) {
            amax = std::max(
                amax,
                std::abs(activation[static_cast<size_t>(block * QK8_1 + k)]));
        }
        const float step = amax / 127.0f;
        for (int64_t row = 0; row < kRows; ++row) {
            float l1 = 0.0f;
            const size_t base = static_cast<size_t>(
                (expert * kRows + row) * weights.k + block * QK8_1);
            for (int64_t k = 0; k < QK8_1; ++k) {
                l1 += std::abs(weights.decoded[base + static_cast<size_t>(k)]);
            }
            budget[static_cast<size_t>(row)] += l1 * step;
        }
    }
    return budget;
}

static bool within_one_step_budget(
        const std::vector<float> & mmvq,
        const std::vector<float> & mmq,
        const std::vector<std::vector<float>> & budgets,
        const std::vector<int32_t> & expert_ids,
        const char * label) {
    if (mmvq.size() != static_cast<size_t>(kRows * kExperts) ||
        mmq.size() != static_cast<size_t>(kRows * kMmqWidth)) {
        std::fprintf(stderr, "%s: size mismatch\n", label);
        return false;
    }
    for (int64_t token = 0; token < kMmqWidth; ++token) {
        const int64_t expert = expert_ids[static_cast<size_t>(token)];
        for (int64_t row = 0; row < kRows; ++row) {
            const float left = mmvq[static_cast<size_t>(expert * kRows + row)];
            const float right = mmq[static_cast<size_t>(token * kRows + row)];
            const float delta = std::abs(left - right);
            const float budget = budgets[static_cast<size_t>(expert)]
                                        [static_cast<size_t>(row)];
            if (!(delta <= budget)) {
                std::fprintf(stderr,
                             "%s: token=%lld row=%lld delta=%g budget=%g\n",
                             label, static_cast<long long>(token),
                             static_cast<long long>(row), delta, budget);
                return false;
            }
        }
    }
    return true;
}

static bool run_inexact_family_comparison(ggml_backend_t backend,
                                          const Fixture & dense,
                                          const Fixture & routed) {
    if (dense.k != routed.k) return false;
    const std::vector<float> row = make_inexact_activation_row(dense.k);
    const std::vector<float> one = repeat_activation(row, kMmvqWidth);
    const std::vector<float> wide = repeat_activation(row, kMmqWidth);
    const std::vector<int32_t> dense_ids(
        static_cast<size_t>(kMmqWidth), 0);
    const std::vector<std::vector<float>> dense_budget = {
        one_step_output_budget(dense, row, 0),
        one_step_output_budget(dense, row, 0),
        one_step_output_budget(dense, row, 0),
    };

    std::vector<float> dense_mmvq;
    std::vector<float> dense_mmq;
    bool ok = run_dense(backend, dense, kMmvqWidth, &one, &dense_mmvq) &&
              run_dense(backend, dense, kMmqWidth, &wide, &dense_mmq);
    if (ok) {
        std::vector<float> replicated(static_cast<size_t>(kRows * kExperts));
        for (int64_t expert = 0; expert < kExperts; ++expert) {
            std::copy(dense_mmvq.begin(), dense_mmvq.end(),
                      replicated.begin() +
                          static_cast<std::ptrdiff_t>(expert * kRows));
        }
        ok = within_one_step_budget(
            replicated, dense_mmq, dense_budget, dense_ids,
            "dense non-grid MMVQ/MMQ");
    }

    std::vector<float> routed_mmvq(static_cast<size_t>(kRows * kExperts));
    for (int64_t expert = 0; ok && expert < kExperts; ++expert) {
        const std::vector<int32_t> one_id = {
            static_cast<int32_t>(expert),
        };
        std::vector<float> output;
        ok = run_routed(
            backend, routed, kMmvqWidth, &one, &one_id, &output);
        if (ok) {
            std::copy(output.begin(), output.end(),
                      routed_mmvq.begin() +
                          static_cast<std::ptrdiff_t>(expert * kRows));
        }
    }
    std::vector<int32_t> routed_ids(static_cast<size_t>(kMmqWidth));
    for (int64_t token = 0; token < kMmqWidth; ++token) {
        routed_ids[static_cast<size_t>(token)] =
            static_cast<int32_t>(token % kExperts);
    }
    std::vector<float> routed_mmq;
    ok = ok && run_routed(
        backend, routed, kMmqWidth, &wide, &routed_ids, &routed_mmq);
    std::vector<std::vector<float>> routed_budget;
    for (int64_t expert = 0; expert < kExperts; ++expert) {
        routed_budget.push_back(
            one_step_output_budget(routed, row, expert));
    }
    return ok && within_one_step_budget(
        routed_mmvq, routed_mmq, routed_budget, routed_ids,
        "routed non-grid MMVQ/MMQ");
}

static void check_k(bool condition, const char * message, int64_t k_dim) {
    if (condition) {
        ++g_pass;
    } else {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s (K=%lld)\n", message,
                     static_cast<long long>(k_dim));
    }
}

} // namespace

int main() {
    const char * enabled = std::getenv("DFLASH_ROCMI4_OPERATOR_ORACLE_HIP");
    if (!enabled || std::strcmp(enabled, "1") != 0) {
        std::printf("rocmi4 operator oracle: skipped (HIP opt-in is off)\n");
        return 0;
    }

    // The dispatch selectors cache their environment on first use.
    setenv("LUCE_MMVQ_MAX_NCOLS", "5", 1);
    setenv("DFLASH_ROCMI4_W4A8_DISPATCH_EVIDENCE", "1", 1);
    setenv("DFLASH_MMID_TELEMETRY", "1", 1);

    ggml_backend_t backend =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    CHECK(backend != nullptr, "HIP backend initializes");
    if (!backend) return 1;

    const enum ggml_backend_dev_type device_type =
        ggml_backend_dev_type(ggml_backend_get_device(backend));
    CHECK(device_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
              device_type == GGML_BACKEND_DEVICE_TYPE_IGPU,
          "oracle runs on a GPU device, not CPU fallback");

    // ROCMI4 MMQ advances K in eight QK32 blocks. The shipped partial cases
    // end with 5, 2, and 4 blocks respectively; K=256 is the full-step control.
    for (const int64_t k_dim : kExactKDims) {
        const Fixture dense = make_weights(1, k_dim);
        const Fixture routed = make_weights(kExperts, k_dim);
        check_k(run_device_dequant(backend, dense),
                "convert.cu dequantizer exactly matches CPU ROCMI4", k_dim);
        check_k(run_dense(backend, dense, kMmvqWidth),
                "dense MMVQ exactly matches CPU F32 matmul", k_dim);
        check_k(run_dense(backend, dense, kMmqWidth),
                "dense MMQ exactly matches CPU F32 matmul", k_dim);
        check_k(run_routed(backend, routed, kMmvqWidth),
                "routed MMVQ exactly matches CPU F32 matmul", k_dim);
        check_k(run_routed(backend, routed, kMmqWidth),
                "routed MMQ exactly matches CPU F32 matmul", k_dim);
    }

    const Fixture dense_control = make_weights(1, kFullIterationK);
    const Fixture routed_control = make_weights(
        kExperts, kFullIterationK);
    CHECK(run_inexact_family_comparison(
              backend, dense_control, routed_control),
          "non-grid MMVQ and MMQ agree within one Q8_1 activation step");

    ggml_backend_free(backend);
    std::printf("rocmi4 operator oracle: %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
