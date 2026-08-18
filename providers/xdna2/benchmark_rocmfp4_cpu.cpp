// Trained DSpark routed-expert CPU placement benchmark.
//
// The complete heterogeneous drafter cannot assume that a worker-owned XRT
// job may call HIP while another resident session is verifying. This probe
// measures the alternative: execute the routed ROCMFP4 gate/up pairs and
// their dependent down projections on the CPU. It intentionally
// validates against Ember's byte-exact scalar reference decoder, dispatches
// the measured work through the production-candidate host decoder, and
// parallelizes only independent projections.  Thirty one-row experts model
// the conservative case where all six routes for five draft rows are unique.

#include "rocmfp4_model_weights.h"
#include "rocmfp4_pack.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kTokens = 5;
constexpr int kEmbd = 4096;
constexpr int kFf = 2048;
constexpr float kClamp = 10.0f;
using Clock = std::chrono::steady_clock;

template <typename Function>
void parallel_for(int count, int threads, const Function & function) {
    std::atomic<int> next{0};
    const int workers = std::max(1, std::min(count, threads));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(workers - 1));
    auto run = [&] {
        for (;;) {
            const int index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= count) return;
            function(index);
        }
    };
    for (int worker = 1; worker < workers; ++worker) pool.emplace_back(run);
    run();
    for (std::thread & worker : pool) worker.join();
}

int cpu_threads() {
    int result = static_cast<int>(std::thread::hardware_concurrency());
    if (result < 1) result = 1;
    if (const char * raw = std::getenv("EMBER_DSPARK_CPU_THREADS")) {
        char * end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw && *end == '\0' && parsed >= 1 && parsed <= 64)
            result = static_cast<int>(parsed);
    }
    return result;
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s DRAFT.gguf [LAYER [FIRST_EXPERT [REPEATS "
                     "[EXPERT_COUNT [ROWS_PER_EXPERT]]]]]\n",
                     argv[0]);
        return 2;
    }
    try {
        const int layer = argc >= 3 ? std::atoi(argv[2]) : 0;
        const int expert_id = argc >= 4 ? std::atoi(argv[3]) : 0;
        const int repeats = argc >= 5 ? std::atoi(argv[4]) : 1;
        const int expert_count = argc >= 6 ? std::atoi(argv[5]) : 1;
        const int rows_per_expert = argc >= 7 ? std::atoi(argv[6]) : kTokens;
        if (layer < 0 || layer >= ember::xdna2::kRocmfp4ModelLayers ||
            expert_id < 0 ||
            expert_id >= ember::xdna2::kRocmfp4ModelExperts ||
            expert_count < 1 || expert_count > 30 ||
            rows_per_expert < 1 || rows_per_expert > kTokens ||
            expert_id + expert_count > ember::xdna2::kRocmfp4ModelExperts ||
            repeats < 1 || repeats > 1000) {
            throw std::runtime_error(
                "layer/expert/repeats/expert-count is out of range");
        }

        std::vector<ember::xdna2::Rocmfp4ModelExpert> experts;
        std::vector<int> expert_ids;
        expert_ids.reserve(static_cast<size_t>(expert_count));
        for (int index = 0; index < expert_count; ++index)
            expert_ids.push_back(expert_id + index);
        std::string error;
        const auto load_begin = Clock::now();
        if (!ember::xdna2::load_rocmfp4_model_experts(
                argv[1], layer, expert_ids, experts, &error)) {
            throw std::runtime_error(error);
        }
        const auto load_end = Clock::now();
        const int rows = rows_per_expert * expert_count;

        std::vector<float> input(static_cast<size_t>(rows) * kEmbd);
        uint32_t state = 0x6d2b79f5u;
        for (float & value : input) {
            state = state * 1664525u + 1013904223u;
            value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f) -
                    1.0f;
        }
        std::vector<float> gate(static_cast<size_t>(rows) * kFf);
        std::vector<float> up(static_cast<size_t>(rows) * kFf);
        std::vector<float> hidden(static_cast<size_t>(rows) * kFf);
        std::vector<float> output(static_cast<size_t>(rows) * kEmbd);
        const int threads = cpu_threads();

        double gate_up_total = 0.0;
        double down_total = 0.0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            const auto gate_up_begin = Clock::now();
            std::atomic<bool> projection_ok{true};
            parallel_for(rows * 2, threads, [&](int task) {
                const int row = task / 2;
                const bool gate_projection = (task & 1) == 0;
                const auto & expert = experts[
                    static_cast<size_t>(row / rows_per_expert)];
                const auto & raw = gate_projection ? expert.gate : expert.up;
                float * destination = (gate_projection ? gate.data() : up.data()) +
                    static_cast<size_t>(row) * kFf;
                if (!ember::xdna2::rocmfp4_gemm_cpu(
                        raw.data(), raw.size(),
                        input.data() + static_cast<size_t>(row) * kEmbd,
                        kEmbd, kFf, 1.0f, destination)) {
                    projection_ok.store(false, std::memory_order_relaxed);
                }
            });
            if (!projection_ok.load(std::memory_order_relaxed))
                throw std::runtime_error("gate/up projection failed");
            parallel_for(rows * kFf, threads, [&](int index) {
                const float gate_value = std::min(gate[static_cast<size_t>(index)],
                                                  kClamp);
                const float up_value = std::max(-kClamp,
                    std::min(up[static_cast<size_t>(index)], kClamp));
                hidden[static_cast<size_t>(index)] =
                    (gate_value / (1.0f + std::exp(-gate_value))) * up_value;
            });
            const auto gate_up_end = Clock::now();
            parallel_for(rows, threads, [&](int row) {
                const auto & expert = experts[
                    static_cast<size_t>(row / rows_per_expert)];
                if (!ember::xdna2::rocmfp4_gemm_cpu(
                        expert.down.data(), expert.down.size(),
                        hidden.data() + static_cast<size_t>(row) * kFf,
                        kFf, kEmbd, 1.0f,
                        output.data() + static_cast<size_t>(row) * kEmbd)) {
                    projection_ok.store(false, std::memory_order_relaxed);
                }
            });
            if (!projection_ok.load(std::memory_order_relaxed))
                throw std::runtime_error("down projection failed");
            const auto down_end = Clock::now();
            gate_up_total += elapsed_ms(gate_up_begin, gate_up_end);
            down_total += elapsed_ms(gate_up_end, down_end);
        }

        double checksum = 0.0;
        bool finite = true;
        for (float value : output) {
            finite = finite && std::isfinite(value);
            checksum += value;
        }
        if (!finite) throw std::runtime_error("non-finite routed-expert output");
        std::vector<float> scalar_gate(kFf), scalar_up(kFf), scalar_hidden(kFf);
        std::vector<float> scalar_output(kEmbd);
        const auto & first = experts.front();
        if (!ember::xdna2::rocmfp4_gemm_raw_reference(
                first.gate.data(), first.gate.size(), input.data(),
                kEmbd, kFf, 1.0f, scalar_gate.data()) ||
            !ember::xdna2::rocmfp4_gemm_raw_reference(
                first.up.data(), first.up.size(), input.data(),
                kEmbd, kFf, 1.0f, scalar_up.data())) {
            throw std::runtime_error("trained scalar gate/up validation failed");
        }
        for (int lane = 0; lane < kFf; ++lane) {
            const float gate_value = std::min(
                scalar_gate[static_cast<size_t>(lane)], kClamp);
            const float up_value = std::max(-kClamp,
                std::min(scalar_up[static_cast<size_t>(lane)], kClamp));
            scalar_hidden[static_cast<size_t>(lane)] =
                (gate_value / (1.0f + std::exp(-gate_value))) * up_value;
        }
        if (!ember::xdna2::rocmfp4_gemm_raw_reference(
                first.down.data(), first.down.size(), scalar_hidden.data(),
                kFf, kEmbd, 1.0f, scalar_output.data())) {
            throw std::runtime_error("trained scalar down validation failed");
        }
        float max_abs = 0.0f;
        double dot = 0.0, actual_square = 0.0, scalar_square = 0.0;
        for (int lane = 0; lane < kEmbd; ++lane) {
            const float actual = output[static_cast<size_t>(lane)];
            const float expected = scalar_output[static_cast<size_t>(lane)];
            max_abs = std::max(max_abs, std::fabs(actual - expected));
            dot += static_cast<double>(actual) * expected;
            actual_square += static_cast<double>(actual) * actual;
            scalar_square += static_cast<double>(expected) * expected;
        }
        const double cosine = dot / std::sqrt(actual_square * scalar_square);
        if (max_abs > 2.0e-3f || cosine < 0.99999999) {
            throw std::runtime_error(
                "AVX-512 trained output exceeds scalar tolerance");
        }
        const double gate_up_ms = gate_up_total / repeats;
        const double down_ms = down_total / repeats;
        std::printf("layer=%d first_expert=%d experts=%d rows_per_expert=%d "
                    "rows=%d threads=%d repeats=%d "
                    "load_ms=%.3f gate_up_swiglu_ms=%.3f down_ms=%.3f "
                    "total_ms=%.3f max_abs=%.9g cosine=%.10f "
                    "checksum=%.9g\n",
                    layer, expert_id, expert_count, rows_per_expert, rows,
                    threads, repeats,
                    elapsed_ms(load_begin, load_end), gate_up_ms, down_ms,
                    gate_up_ms + down_ms, max_abs, cosine, checksum);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "DSPARK_ROCMFP4_CPU_BENCH_ERROR: %s\n",
                     exception.what());
        return 1;
    }
}
