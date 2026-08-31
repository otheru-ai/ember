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
// leave partial MMQ_ITER_K=256 iterations and the shipped output N dimensions
// that require a partial MMQ row tile, each beside a full-tile control.  A
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
#include <string>
#include <unistd.h>
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
constexpr int64_t kFullRowTileN = 128;
constexpr int64_t kExperts = 3;
constexpr int64_t kMmvqWidth = 1;
constexpr int64_t kMmqWidth = 16;
constexpr int64_t kMaxMmqRowTile = 128;
constexpr float kActivationScale = 0.25f;
constexpr float kMaxAbsBudget = 0.0f;
constexpr int64_t kExactKDims[] = {kFullIterationK, 160, 320, 640};
constexpr int64_t kExactNDims[] = {4, 48, kFullRowTileN, 320};

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
    int64_t rows = 0;
    std::vector<block_rocmi4> packed;
    std::vector<float> decoded;
};

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        stream_ = std::tmpfile();
        if (!stream_) return;
        saved_fd_ = dup(fileno(stderr));
        if (saved_fd_ < 0 || dup2(fileno(stream_), fileno(stderr)) < 0) {
            if (saved_fd_ >= 0) close(saved_fd_);
            saved_fd_ = -1;
            std::fclose(stream_);
            stream_ = nullptr;
            return;
        }
        active_ = true;
    }

    StderrCapture(const StderrCapture &) = delete;
    StderrCapture & operator=(const StderrCapture &) = delete;

    ~StderrCapture() {
        if (active_) (void) finish();
        if (stream_) std::fclose(stream_);
    }

    bool ready() const { return active_; }

    std::string finish() {
        if (!active_) return {};
        std::fflush(stderr);
        (void) dup2(saved_fd_, fileno(stderr));
        close(saved_fd_);
        saved_fd_ = -1;
        active_ = false;
        std::rewind(stream_);
        std::string output;
        char buffer[4096];
        for (;;) {
            const size_t count = std::fread(buffer, 1, sizeof(buffer), stream_);
            output.append(buffer, count);
            if (count < sizeof(buffer)) break;
        }
        std::fclose(stream_);
        stream_ = nullptr;
        return output;
    }

private:
    FILE * stream_ = nullptr;
    int saved_fd_ = -1;
    bool active_ = false;
};

static bool dispatch_line_has(const std::string & log,
                              const std::string & event,
                              const std::string & op,
                              int64_t width,
                              const std::string & path,
                              const std::string & weight) {
    const std::string width_field =
        "physical_q=" + std::to_string(width);
    size_t start = 0;
    while (start < log.size()) {
        const size_t end = log.find('\n', start);
        const size_t count = end == std::string::npos
            ? log.size() - start : end - start;
        const std::string line = log.substr(start, count);
        if (line.find("[rocmi4-w4a8-dispatch]") != std::string::npos &&
            line.find("event=" + event) != std::string::npos &&
            line.find("op=" + op) != std::string::npos &&
            line.find(width_field) != std::string::npos &&
            (path.empty() ||
             line.find("path=" + path) != std::string::npos) &&
            line.find("weight=" + weight) != std::string::npos) {
            return true;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}

static bool verify_dispatch(const std::string & log, const char * op,
                            int64_t width, const char * path,
                            const char * weight) {
    // Preserve the route evidence in the external run log after capturing it
    // for the in-process assertion.
    if (!log.empty()) {
        (void) std::fwrite(log.data(), 1, log.size(), stderr);
        std::fflush(stderr);
    }
    const bool route = dispatch_line_has(
        log, "route", op, width, path, weight);
    const bool kernel = std::strcmp(path, "mmq") != 0 || dispatch_line_has(
        log, "kernel", op, width, "", weight);
    if (!route || !kernel) {
        std::fprintf(stderr,
                     "dispatch proof missing: op=%s physical_q=%lld "
                     "path=%s weight=%s route=%d kernel=%d\n",
                     op, static_cast<long long>(width), path, weight,
                     route ? 1 : 0, kernel ? 1 : 0);
    }
    return route && kernel;
}

static int signed_nibble(int64_t expert, int64_t row, int64_t block,
                         int64_t index, int salt) {
    const int value = static_cast<int>(
        (expert * 11 + row * 7 + block * 5 + index * salt) & 15);
    return value < 8 ? value : value - 16;
}

static Fixture make_weights(int64_t experts, int64_t k, int64_t rows) {
    const int64_t blocks_per_row = k / QK_ROCMI4;
    Fixture fixture;
    fixture.k = k;
    fixture.rows = rows;
    fixture.packed.resize(static_cast<size_t>(
        experts * rows * blocks_per_row));
    fixture.decoded.resize(static_cast<size_t>(experts * rows * k));

    constexpr uint8_t scales[] = {0x30, 0x38, 0x40}; // 0.25, 0.5, 1.0
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < rows; ++row) {
            block_rocmi4 * packed_row = fixture.packed.data() +
                (expert * rows + row) * blocks_per_row;
            float * decoded_row = fixture.decoded.data() +
                (expert * rows + row) * k;
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
    std::vector<float> output(
        static_cast<size_t>(weights.rows * width), 0.0f);
    for (int64_t token = 0; token < width; ++token) {
        for (int64_t row = 0; row < weights.rows; ++row) {
            float sum = 0.0f;
            for (int64_t k = 0; k < k_dim; ++k) {
                sum += weights.decoded[static_cast<size_t>(row * k_dim + k)] *
                       input[static_cast<size_t>(token * k_dim + k)];
            }
            output[static_cast<size_t>(token * weights.rows + row)] = sum;
        }
    }
    return output;
}

static std::vector<float> routed_reference(const Fixture & weights,
                                           const std::vector<float> & input,
                                           const std::vector<int32_t> & ids,
                                           int64_t width) {
    const int64_t k_dim = weights.k;
    std::vector<float> output(
        static_cast<size_t>(weights.rows * width), 0.0f);
    for (int64_t token = 0; token < width; ++token) {
        const int64_t expert = ids[static_cast<size_t>(token)];
        for (int64_t row = 0; row < weights.rows; ++row) {
            float sum = 0.0f;
            const size_t weight_base = static_cast<size_t>(
                (expert * weights.rows + row) * k_dim);
            for (int64_t k = 0; k < k_dim; ++k) {
                sum += weights.decoded[weight_base + static_cast<size_t>(k)] *
                       input[static_cast<size_t>(token * k_dim + k)];
            }
            output[static_cast<size_t>(token * weights.rows + row)] = sum;
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

static ggml_backend_buffer_t allocate_with_output_guard(
        ggml_backend_t backend, const std::vector<ggml_tensor *> & tensors,
        ggml_tensor * output, ggml_tensor * guard) {
    ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    std::vector<size_t> offsets;
    offsets.reserve(tensors.size());
    size_t total = 0;
    for (const ggml_tensor * tensor : tensors) {
        total = GGML_PAD(total, alignment);
        offsets.push_back(total);
        total += ggml_backend_buft_get_alloc_size(buft, tensor);
    }
    const auto output_it = std::find(
        tensors.begin(), tensors.end(), output);
    if (output_it == tensors.end()) return nullptr;
    const size_t output_index = static_cast<size_t>(
        std::distance(tensors.begin(), output_it));
    const size_t guard_offset =
        offsets[output_index] + ggml_nbytes(output);
    if (guard_offset % alignment != 0) return nullptr;
    total = std::max(
        total, guard_offset + ggml_backend_buft_get_alloc_size(buft, guard));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, total);
    if (!buffer) return nullptr;
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (ggml_backend_tensor_alloc(
                buffer, tensors[i], base + offsets[i]) !=
            GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            return nullptr;
        }
    }
    if (ggml_backend_tensor_alloc(
            buffer, guard, base + guard_offset) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        return nullptr;
    }
    return buffer;
}

static size_t output_guard_bytes(int64_t rows, int64_t width) {
    const int64_t padded_rows = GGML_PAD(rows, kMaxMmqRowTile);
    return static_cast<size_t>(
        (padded_rows - rows) * width * static_cast<int64_t>(sizeof(float)));
}

static bool output_guard_unchanged(ggml_tensor * output,
                                   ggml_tensor * guard,
                                   const std::vector<uint8_t> & sentinel,
                                   const char * label) {
    if (!guard) return true;
    std::vector<uint8_t> actual(sentinel.size());
    ggml_backend_tensor_get(guard, actual.data(), 0, actual.size());
    const auto mismatch = std::mismatch(actual.begin(), actual.end(),
                                        sentinel.begin());
    if (mismatch.first == actual.end()) return true;
    const size_t offset = static_cast<size_t>(
        std::distance(actual.begin(), mismatch.first));
    const size_t output_nbytes = ggml_nbytes(output);
    const size_t output_alloc_size =
        ggml_backend_buffer_get_alloc_size(output->buffer, output);
    const bool within_allocation_padding =
        output_nbytes + offset < output_alloc_size;
    std::fprintf(stderr,
                 "%s: output write crossed logical extent at byte=%zu "
                 "output_nbytes=%zu output_alloc_size=%zu scope=%s "
                 "actual=0x%02x expected=0x%02x\n",
                 label, offset, output_nbytes, output_alloc_size,
                 within_allocation_padding
                     ? "within_allocation_padding"
                     : "beyond_output_allocation",
                 static_cast<unsigned>(actual[offset]),
                 static_cast<unsigned>(sentinel[offset]));
    return false;
}

static bool run_device_dequant(ggml_backend_t backend,
                               const Fixture & fixture) {
    ggml_context * ctx = make_context();
    if (!ctx) return false;
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_Q4_0_ROCMI4, fixture.k, fixture.rows);
    ggml_tensor * decoded = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, fixture.k, fixture.rows);
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
        std::snprintf(label, sizeof(label),
                      "convert.cu ROCMI4 K=%lld N=%lld",
                      static_cast<long long>(fixture.k),
                      static_cast<long long>(fixture.rows));
        ok = exact_vectors(actual, fixture.decoded, label);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool run_dense(ggml_backend_t backend, const Fixture & fixture,
                      int64_t width,
                      const std::vector<float> * supplied_activation = nullptr,
                      std::vector<float> * captured = nullptr,
                      bool check_write_extent = false) {
    ggml_context * ctx = make_context();
    if (!ctx) return false;
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_Q4_0_ROCMI4, fixture.k, fixture.rows);
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, fixture.k, width);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(weights, "rocmi4_oracle_dense_weight");
    ggml_set_name(output, "rocmi4_oracle_dense_output");
    const size_t guard_bytes = check_write_extent
        ? output_guard_bytes(fixture.rows, width) : 0;
    ggml_tensor * guard = guard_bytes > 0
        ? ggml_new_tensor_1d(
            ctx, GGML_TYPE_I8, static_cast<int64_t>(guard_bytes))
        : nullptr;
    ggml_backend_buffer_t buffer = guard
        ? allocate_with_output_guard(
            backend, {weights, input, output}, output, guard)
        : ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> activation = supplied_activation
        ? *supplied_activation : make_activations(fixture.k, width);
    const std::vector<float> expected = supplied_activation
        ? std::vector<float>() : dense_reference(fixture, activation, width);
    const std::vector<uint8_t> sentinel(guard_bytes, UINT8_C(0xa5));
    ggml_backend_tensor_set(weights, fixture.packed.data(), 0,
                            fixture.packed.size() * sizeof(block_rocmi4));
    ggml_backend_tensor_set(input, activation.data(), 0,
                            activation.size() * sizeof(float));
    if (guard) {
        ggml_backend_tensor_set(
            guard, sentinel.data(), 0, sentinel.size());
    }
    StderrCapture dispatch_log;
    const bool dispatch_capture_ready = dispatch_log.ready();
    const bool supported = ggml_backend_supports_op(backend, output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = supported
        ? ggml_backend_graph_compute(backend, graph)
        : GGML_STATUS_FAILED;
    std::vector<float> actual(
        static_cast<size_t>(fixture.rows * width));
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(float));
        if (supplied_activation) {
            if (captured) *captured = actual;
        } else {
            char label[64];
            std::snprintf(label, sizeof(label),
                          "dense %s K=%lld N=%lld",
                          width == kMmvqWidth ? "MMVQ" : "MMQ",
                          static_cast<long long>(fixture.k),
                          static_cast<long long>(fixture.rows));
            ok = exact_vectors(
                actual, expected, label);
        }
    }
    ok = output_guard_unchanged(
             output, guard, sentinel, "dense MMQ") && ok;
    const std::string dispatch = dispatch_log.finish();
    const bool dispatch_ok = dispatch_capture_ready && verify_dispatch(
        dispatch, "dense", width,
        width == kMmvqWidth ? "mmvq" : "mmq",
        "rocmi4_oracle_dense_weight");
    ok = ok && dispatch_ok;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool run_routed(ggml_backend_t backend, const Fixture & fixture,
                       int64_t width,
                       const std::vector<float> * supplied_activation = nullptr,
                       const std::vector<int32_t> * supplied_ids = nullptr,
                       std::vector<float> * captured = nullptr,
                       bool check_write_extent = false) {
    ggml_context * ctx = make_context();
    if (!ctx) return false;
    ggml_tensor * weights = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q4_0_ROCMI4, fixture.k, fixture.rows, kExperts);
    ggml_tensor * input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, fixture.k, 1, width);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, 1, width);
    ggml_tensor * output = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_name(weights, "rocmi4_oracle_expert_weight");
    ggml_set_name(output, "rocmi4_oracle_expert_output");
    const size_t guard_bytes = check_write_extent
        ? output_guard_bytes(fixture.rows, width) : 0;
    ggml_tensor * guard = guard_bytes > 0
        ? ggml_new_tensor_1d(
            ctx, GGML_TYPE_I8, static_cast<int64_t>(guard_bytes))
        : nullptr;
    ggml_backend_buffer_t buffer = guard
        ? allocate_with_output_guard(
            backend, {weights, input, ids, output}, output, guard)
        : ggml_backend_alloc_ctx_tensors(ctx, backend);
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
    const std::vector<uint8_t> sentinel(guard_bytes, UINT8_C(0xa5));
    ggml_backend_tensor_set(weights, fixture.packed.data(), 0,
                            fixture.packed.size() * sizeof(block_rocmi4));
    ggml_backend_tensor_set(input, activation.data(), 0,
                            activation.size() * sizeof(float));
    ggml_backend_tensor_set(ids, expert_ids.data(), 0,
                            expert_ids.size() * sizeof(int32_t));
    if (guard) {
        ggml_backend_tensor_set(
            guard, sentinel.data(), 0, sentinel.size());
    }
    StderrCapture dispatch_log;
    const bool dispatch_capture_ready = dispatch_log.ready();
    const bool supported = ggml_backend_supports_op(backend, output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = supported
        ? ggml_backend_graph_compute(backend, graph)
        : GGML_STATUS_FAILED;
    std::vector<float> actual(
        static_cast<size_t>(fixture.rows * width));
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(float));
        if (supplied_activation) {
            if (captured) *captured = actual;
        } else {
            char label[64];
            std::snprintf(label, sizeof(label),
                          "routed %s K=%lld N=%lld",
                          width == kMmvqWidth ? "MMVQ" : "MMQ",
                          static_cast<long long>(fixture.k),
                          static_cast<long long>(fixture.rows));
            ok = exact_vectors(
                actual, expected, label);
        }
    }
    ok = output_guard_unchanged(
             output, guard, sentinel, "routed MMQ") && ok;
    const std::string dispatch = dispatch_log.finish();
    const bool dispatch_ok = dispatch_capture_ready && verify_dispatch(
        dispatch, "routed_expert", width,
        width == kMmvqWidth ? "mmvq" : "mmq",
        "rocmi4_oracle_expert_weight");
    ok = ok && dispatch_ok;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static std::vector<float> one_step_output_budget(
        const Fixture & weights, const std::vector<float> & activation,
        int64_t expert) {
    std::vector<float> budget(static_cast<size_t>(weights.rows), 0.0f);
    for (int64_t block = 0; block < weights.k / QK8_1; ++block) {
        float amax = 0.0f;
        for (int64_t k = 0; k < QK8_1; ++k) {
            amax = std::max(
                amax,
                std::abs(activation[static_cast<size_t>(block * QK8_1 + k)]));
        }
        const float step = amax / 127.0f;
        for (int64_t row = 0; row < weights.rows; ++row) {
            float l1 = 0.0f;
            const size_t base = static_cast<size_t>(
                (expert * weights.rows + row) * weights.k + block * QK8_1);
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
    const int64_t rows = static_cast<int64_t>(budgets.front().size());
    if (mmvq.size() != static_cast<size_t>(rows * kExperts) ||
        mmq.size() != static_cast<size_t>(rows * kMmqWidth)) {
        std::fprintf(stderr, "%s: size mismatch\n", label);
        return false;
    }
    for (int64_t token = 0; token < kMmqWidth; ++token) {
        const int64_t expert = expert_ids[static_cast<size_t>(token)];
        for (int64_t row = 0; row < rows; ++row) {
            const float left = mmvq[static_cast<size_t>(expert * rows + row)];
            const float right = mmq[static_cast<size_t>(token * rows + row)];
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
    if (dense.k != routed.k || dense.rows != routed.rows) return false;
    const int64_t rows = dense.rows;
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
        std::vector<float> replicated(static_cast<size_t>(rows * kExperts));
        for (int64_t expert = 0; expert < kExperts; ++expert) {
            std::copy(dense_mmvq.begin(), dense_mmvq.end(),
                      replicated.begin() +
                          static_cast<std::ptrdiff_t>(expert * rows));
        }
        ok = within_one_step_budget(
            replicated, dense_mmq, dense_budget, dense_ids,
            "dense non-grid MMVQ/MMQ");
    }

    std::vector<float> routed_mmvq(static_cast<size_t>(rows * kExperts));
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
                          static_cast<std::ptrdiff_t>(expert * rows));
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

static void check_shape(bool condition, const char * message,
                        int64_t k_dim, int64_t rows) {
    if (condition) {
        ++g_pass;
    } else {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s (K=%lld N=%lld)\n", message,
                     static_cast<long long>(k_dim),
                     static_cast<long long>(rows));
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
        const Fixture dense = make_weights(1, k_dim, kFullRowTileN);
        const Fixture routed = make_weights(
            kExperts, k_dim, kFullRowTileN);
        check_shape(run_device_dequant(backend, dense),
                    "convert.cu dequantizer exactly matches CPU ROCMI4",
                    k_dim, kFullRowTileN);
        check_shape(run_dense(backend, dense, kMmvqWidth),
                    "dense MMVQ exactly matches CPU F32 matmul",
                    k_dim, kFullRowTileN);
        check_shape(run_dense(backend, dense, kMmqWidth),
                    "dense MMQ exactly matches CPU F32 matmul",
                    k_dim, kFullRowTileN);
        check_shape(run_routed(backend, routed, kMmvqWidth),
                    "routed MMVQ exactly matches CPU F32 matmul",
                    k_dim, kFullRowTileN);
        check_shape(run_routed(backend, routed, kMmqWidth),
                    "routed MMQ exactly matches CPU F32 matmul",
                    k_dim, kFullRowTileN);
    }

    // MMQ selects a separately instantiated need_check=true kernel when the
    // output rows do not fill a tile. N=128 is a full-row control; 4 and 48
    // are guaranteed tail cases for gfx1151's 64/128-row tile variants, while
    // 320 keeps the other shipped small-output shape in the sweep.
    for (const int64_t rows : kExactNDims) {
        const Fixture dense = make_weights(1, kFullIterationK, rows);
        const Fixture routed = make_weights(
            kExperts, kFullIterationK, rows);
        check_shape(run_device_dequant(backend, dense),
                    "convert.cu dequantizer exactly matches CPU ROCMI4",
                    kFullIterationK, rows);
        check_shape(run_dense(backend, dense, kMmvqWidth),
                    "dense MMVQ exactly matches CPU F32 matmul",
                    kFullIterationK, rows);
        check_shape(run_dense(
                        backend, dense, kMmqWidth, nullptr, nullptr, true),
                    "dense MMQ exactly matches CPU F32 matmul",
                    kFullIterationK, rows);
        check_shape(run_routed(backend, routed, kMmvqWidth),
                    "routed MMVQ exactly matches CPU F32 matmul",
                    kFullIterationK, rows);
        check_shape(run_routed(
                        backend, routed, kMmqWidth,
                        nullptr, nullptr, nullptr, true),
                    "routed MMQ exactly matches CPU F32 matmul",
                    kFullIterationK, rows);
    }

    const Fixture dense_control = make_weights(
        1, kFullIterationK, kFullRowTileN);
    const Fixture routed_control = make_weights(
        kExperts, kFullIterationK, kFullRowTileN);
    CHECK(run_inexact_family_comparison(
              backend, dense_control, routed_control),
          "non-grid MMVQ and MMQ agree within one Q8_1 activation step");

    ggml_backend_free(backend);
    std::printf("rocmi4 operator oracle: %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
