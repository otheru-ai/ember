// Measure the cost of alternating between two full-array XDNA2 overlays.
// Standalone kernel timings cannot expose temporal-partition reload overhead;
// a DSpark provider changes projection shape repeatedly within every layer.

#include "q8_0_pack.h"
#include "q8_model_weights.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBatch = 5;
constexpr int kK = 4096;
constexpr int kQaN = 1024;
constexpr int kKvN = 512;
using Clock = std::chrono::steady_clock;

std::vector<uint32_t> read_instructions(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path);
    const std::streamsize length = file.tellg();
    if (length <= 0 || length % static_cast<std::streamsize>(sizeof(uint32_t)))
        throw std::runtime_error("invalid instruction stream " + path);
    file.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(length) / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char *>(words.data()), length))
        throw std::runtime_error("short instruction stream " + path);
    return words;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

class Projection {
public:
    Projection(xrt::device & device, const std::string & artifact_base,
               int k, int n, const std::vector<uint8_t> & packed)
        : k_(k), n_(n),
          rows_(ember::xdna2::q8_projection_rows(n)),
          groups_(n / (rows_ * ember::xdna2::kQ8OutputsPerRow)),
          xclbin_(artifact_base + ".xclbin") {
        const auto kernels = xclbin_.get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & kernel) {
                return kernel.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) throw std::runtime_error("AIE kernel absent");
        device.register_xclbin(xclbin_);
        context_ = std::make_unique<xrt::hw_context>(device, xclbin_.get_uuid());
        kernel_ = std::make_unique<xrt::kernel>(*context_, found->get_name());
        instructions_ = read_instructions(artifact_base + ".insts");
        instruction_bo_ = std::make_unique<xrt::bo>(
            device, instructions_.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, kernel_->group_id(1));
        std::memcpy(instruction_bo_->map<void *>(), instructions_.data(),
                    instructions_.size() * sizeof(uint32_t));
        instruction_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        input_bo_ = std::make_unique<xrt::bo>(
            device, static_cast<size_t>(groups_) * kBatch * k_ * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(3));
        weight_bo_ = std::make_unique<xrt::bo>(
            device, packed.size(), XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(4));
        output_bo_ = std::make_unique<xrt::bo>(
            device, static_cast<size_t>(kBatch) * n_ * sizeof(float),
            XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(5));
        dummy6_ = std::make_unique<xrt::bo>(
            device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(6));
        dummy7_ = std::make_unique<xrt::bo>(
            device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(7));
        uint16_t * input = input_bo_->map<uint16_t *>();
        for (int group = 0; group < groups_; ++group) {
            for (int token = 0; token < kBatch; ++token) {
                for (int lane = 0; lane < k_; ++lane) {
                    const size_t index =
                        (static_cast<size_t>(group) * kBatch +
                         static_cast<size_t>(token)) * static_cast<size_t>(k_) +
                        static_cast<size_t>(lane);
                    input[index] = float_to_bf16(
                        static_cast<float>((lane + token) % 31 - 15) / 32.0f);
                }
            }
        }
        std::memcpy(weight_bo_->map<void *>(), packed.data(), packed.size());
        input_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weight_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void run() {
        xrt::run command(*kernel_);
        command.set_arg(0, 3);
        command.set_arg(1, *instruction_bo_);
        command.set_arg(2, static_cast<uint32_t>(instructions_.size()));
        command.set_arg(3, *input_bo_);
        command.set_arg(4, *weight_bo_);
        command.set_arg(5, *output_bo_);
        command.set_arg(6, *dummy6_);
        command.set_arg(7, *dummy7_);
        command.start();
        if (command.wait() != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("projection command failed");
        output_bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

private:
    int k_;
    int n_;
    int rows_;
    int groups_;
    xrt::xclbin xclbin_;
    std::unique_ptr<xrt::hw_context> context_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::vector<uint32_t> instructions_;
    std::unique_ptr<xrt::bo> instruction_bo_;
    std::unique_ptr<xrt::bo> input_bo_;
    std::unique_ptr<xrt::bo> weight_bo_;
    std::unique_ptr<xrt::bo> output_bo_;
    std::unique_ptr<xrt::bo> dummy6_;
    std::unique_ptr<xrt::bo> dummy7_;
};

double run_repeated(Projection & projection, int repeats) {
    const auto begin = Clock::now();
    for (int run = 0; run < repeats; ++run) projection.run();
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count() /
           static_cast<double>(repeats);
}

class FixedOverlay {
public:
    struct Workload {
        Workload(xrt::device & device, xrt::kernel & kernel,
                 const std::vector<uint8_t> & packed)
            : input(device, static_cast<size_t>(kBatch) * kK * sizeof(uint16_t),
                    XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3)),
              weight(device, packed.size(), XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(4)),
              output(device, static_cast<size_t>(kBatch) * kQaN * sizeof(float),
                     XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5)) {
            uint16_t * mapped_input = input.map<uint16_t *>();
            for (int token = 0; token < kBatch; ++token) {
                for (int lane = 0; lane < kK; ++lane) {
                    mapped_input[static_cast<size_t>(token) * kK + lane] =
                        float_to_bf16(static_cast<float>(
                            (lane + token) % 31 - 15) / 32.0f);
                }
            }
            std::memcpy(weight.map<void *>(), packed.data(), packed.size());
            input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        xrt::bo input;
        xrt::bo weight;
        xrt::bo output;
    };

    FixedOverlay(xrt::device & device, const std::string & artifact_base,
                 const std::vector<uint8_t> & qa,
                 const std::vector<uint8_t> & padded_kv)
        : xclbin_(artifact_base + ".xclbin") {
        const auto kernels = xclbin_.get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & kernel) {
                return kernel.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) throw std::runtime_error("AIE kernel absent");
        device.register_xclbin(xclbin_);
        context_ = std::make_unique<xrt::hw_context>(device, xclbin_.get_uuid());
        kernel_ = std::make_unique<xrt::kernel>(*context_, found->get_name());
        instructions_ = read_instructions(artifact_base + ".insts");
        instruction_bo_ = std::make_unique<xrt::bo>(
            device, instructions_.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, kernel_->group_id(1));
        std::memcpy(instruction_bo_->map<void *>(), instructions_.data(),
                    instructions_.size() * sizeof(uint32_t));
        instruction_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        dummy6_ = std::make_unique<xrt::bo>(
            device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(6));
        dummy7_ = std::make_unique<xrt::bo>(
            device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(7));
        qa_ = std::make_unique<Workload>(device, *kernel_, qa);
        kv_ = std::make_unique<Workload>(device, *kernel_, padded_kv);
    }

    Workload & qa() { return *qa_; }
    Workload & kv() { return *kv_; }

    void run(Workload & workload) {
        xrt::run command(*kernel_);
        command.set_arg(0, 3);
        command.set_arg(1, *instruction_bo_);
        command.set_arg(2, static_cast<uint32_t>(instructions_.size()));
        command.set_arg(3, workload.input);
        command.set_arg(4, workload.weight);
        command.set_arg(5, workload.output);
        command.set_arg(6, *dummy6_);
        command.set_arg(7, *dummy7_);
        command.start();
        if (command.wait() != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("fixed-overlay command failed");
        workload.output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

private:
    xrt::xclbin xclbin_;
    std::unique_ptr<xrt::hw_context> context_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::vector<uint32_t> instructions_;
    std::unique_ptr<xrt::bo> instruction_bo_;
    std::unique_ptr<xrt::bo> dummy6_;
    std::unique_ptr<xrt::bo> dummy7_;
    std::unique_ptr<Workload> qa_;
    std::unique_ptr<Workload> kv_;
};

double run_repeated(FixedOverlay & overlay, FixedOverlay::Workload & workload,
                    int repeats) {
    const auto begin = Clock::now();
    for (int run = 0; run < repeats; ++run) overlay.run(workload);
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count() /
           static_cast<double>(repeats);
}

double validate_padded_kv(FixedOverlay::Workload & workload,
                          const std::vector<uint8_t> & raw) {
    const float * staging = workload.output.map<const float *>();
    std::vector<float> input(static_cast<size_t>(kBatch) * kK);
    std::vector<float> expected(static_cast<size_t>(kBatch) * kKvN);
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (int token = 0; token < kBatch; ++token) {
        for (int lane = 0; lane < kK; ++lane) {
            const uint16_t bits = float_to_bf16(
                static_cast<float>((lane + token) % 31 - 15) / 32.0f);
            const uint32_t fp32_bits = static_cast<uint32_t>(bits) << 16;
            std::memcpy(&input[static_cast<size_t>(token) * kK + lane],
                        &fp32_bits, sizeof(fp32_bits));
        }
        if (!ember::xdna2::q8_gemm_raw_reference(
                raw.data(), raw.size(),
                input.data() + static_cast<size_t>(token) * kK,
                kK, kKvN,
                expected.data() + static_cast<size_t>(token) * kKvN))
            throw std::runtime_error("padded KV reference failed");
        for (int output = 0; output < kKvN; ++output) {
            const int column = output / ember::xdna2::kQ8TileN;
            const int lane = output % ember::xdna2::kQ8TileN;
            const size_t source =
                static_cast<size_t>(column) * kBatch *
                    ember::xdna2::kQ8TileN +
                static_cast<size_t>(token) * ember::xdna2::kQ8TileN + lane;
            const double actual = staging[source];
            const double reference =
                expected[static_cast<size_t>(token) * kKvN + output];
            dot += actual * reference;
            actual_norm += actual * actual;
            expected_norm += reference * reference;
        }
    }
    return dot / std::sqrt(actual_norm * expected_norm);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: %s ARTIFACT_DIR DRAFT_GGUF [REPEATS]\n", argv[0]);
        return 2;
    }
    try {
        int repeats = 20;
        if (argc == 4) {
            char * end = nullptr;
            const long parsed = std::strtol(argv[3], &end, 10);
            if (end == argv[3] || *end != '\0' || parsed < 2 || parsed > 1000)
                throw std::runtime_error("repeats must be in [2,1000]");
            repeats = static_cast<int>(parsed);
        }
        ember::xdna2::Q8ModelProjection qa;
        ember::xdna2::Q8ModelProjection kv;
        std::string error;
        if (!ember::xdna2::load_q8_model_projection(
                argv[2], "blk.0.attn_q_a.weight", 4096, 1024, qa, &error) ||
            !ember::xdna2::load_q8_model_projection(
                argv[2], "blk.0.attn_kv.weight", 4096, 512, kv, &error))
            throw std::runtime_error(error);
        const std::string directory = argv[1];
        xrt::device device(0);
        Projection qa_program(device,
            directory + "/q8_projection_v3_4096x1024_b5", 4096, 1024,
            qa.packed);
        Projection kv_program(device,
            directory + "/q8_projection_v3_4096x512_b5", 4096, 512,
            kv.packed);
        qa_program.run();
        kv_program.run();
        const double qa_ms = run_repeated(qa_program, repeats);
        const double kv_ms = run_repeated(kv_program, repeats);
        const auto begin = Clock::now();
        for (int run = 0; run < repeats; ++run) {
            qa_program.run();
            kv_program.run();
        }
        const double alternate_pair_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - begin).count() /
            static_cast<double>(repeats);
        std::printf(
            "xdna_overlay_switch repeats=%d qa_ms=%.6f kv_ms=%.6f "
            "alternate_pair_ms=%.6f switch_overhead_ms=%.6f\n",
            repeats, qa_ms, kv_ms, alternate_pair_ms,
            alternate_pair_ms - qa_ms - kv_ms);

        std::vector<uint8_t> padded_kv;
        if (!ember::xdna2::pad_q8_projection_rows(
                kv.packed, kK, kKvN, kQaN, padded_kv, &error))
            throw std::runtime_error(error);
        FixedOverlay fixed(device,
            directory + "/q8_projection_v3_4096x1024_b5",
            qa.packed, padded_kv);
        fixed.run(fixed.qa());
        fixed.run(fixed.kv());
        const double padded_kv_cosine = validate_padded_kv(fixed.kv(), kv.raw);
        const double fixed_qa_ms = run_repeated(fixed, fixed.qa(), repeats);
        const double fixed_kv_ms = run_repeated(fixed, fixed.kv(), repeats);
        const auto fixed_begin = Clock::now();
        for (int run = 0; run < repeats; ++run) {
            fixed.run(fixed.qa());
            fixed.run(fixed.kv());
        }
        const double fixed_pair_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - fixed_begin).count() /
            static_cast<double>(repeats);
        std::printf(
            "xdna_fixed_overlay repeats=%d qa_ms=%.6f padded_kv_ms=%.6f "
            "alternate_pair_ms=%.6f pair_overhead_ms=%.6f "
            "padded_kv_cosine=%.10f\n",
            repeats, fixed_qa_ms, fixed_kv_ms, fixed_pair_ms,
            fixed_pair_ms - fixed_qa_ms - fixed_kv_ms, padded_kv_cosine);
        if (padded_kv_cosine < 0.99999)
            throw std::runtime_error("padded KV projection failed accuracy gate");
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "XDNA overlay-switch benchmark failed: %s\n",
                     exception.what());
        return 1;
    }
}
