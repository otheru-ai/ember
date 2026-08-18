// Validate all trained DSpark Q8 projections through one resident AIE overlay.
// Shape-specific instruction BOs may change shim DMA descriptors, but this
// process registers exactly one xclbin and creates exactly one hw_context.

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
#include <utility>
#include <vector>

namespace {

constexpr int kBatch = 5;
constexpr int kHeaderElements = kBatch * ember::xdna2::kQ8TileK;
constexpr int kOutputsPerGroup = ember::xdna2::kQ8OutputsPerPass;
using Clock = std::chrono::steady_clock;

struct Shape {
    const char * label;
    const char * tensor;
    int k;
    int n;
    int instruction_n;
};

constexpr Shape kShapes[] = {
    {"qakv", nullptr, 4096, 1536, 1024},
    {"qb", "blk.0.attn_q_b.weight", 1024, 32768, 32768},
    {"oa", "blk.0.attn_output_a.weight", 4096, 8192, 8192},
    {"ob", "blk.0.attn_output_b.weight", 8192, 4096, 4096},
};

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

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::string instruction_path(const std::string & directory, const Shape & shape) {
    return directory + "/q8_projection_universal_v4_" +
           std::to_string(shape.k) + "x" +
           std::to_string(shape.instruction_n) +
           "_b5.insts";
}

struct Metrics {
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    double cosine = 0.0;
};

class UniversalOverlay {
public:
    class Workload {
    public:
        Workload(xrt::device & device, xrt::kernel & kernel,
                 const std::string & directory, const std::string & model,
                 const Shape & shape)
            : shape_(shape),
              padded_n_(std::max(shape.n, kOutputsPerGroup)),
              groups_(padded_n_ / kOutputsPerGroup) {
            std::string error;
            if (shape.tensor) {
                if (!ember::xdna2::load_q8_model_projection(
                        model.c_str(), shape.tensor, shape.k, shape.n,
                        projection_, &error))
                    throw std::runtime_error(error);
            } else {
                ember::xdna2::Q8ModelProjection qa;
                ember::xdna2::Q8ModelProjection kv;
                if (!ember::xdna2::load_q8_model_projection(
                        model.c_str(), "blk.0.attn_q_a.weight", shape.k,
                        1024, qa, &error) ||
                    !ember::xdna2::load_q8_model_projection(
                        model.c_str(), "blk.0.attn_kv.weight", shape.k,
                        512, kv, &error))
                    throw std::runtime_error(error);
                projection_.raw = std::move(qa.raw);
                projection_.raw.insert(projection_.raw.end(),
                                       kv.raw.begin(), kv.raw.end());
                if (!ember::xdna2::concat_q8_projection_rows(
                        qa.packed, shape.k, 1024, kv.packed, 512,
                        projection_.packed, &error))
                    throw std::runtime_error(error);
            }
            if (shape.n < kOutputsPerGroup) {
                if (!ember::xdna2::pad_q8_projection_rows(
                        projection_.packed, shape.k, shape.n, padded_n_,
                        packed_, &error))
                    throw std::runtime_error(error);
            } else {
                packed_ = projection_.packed;
            }
            instructions_ = read_instructions(instruction_path(directory, shape));
            instruction_bo_ = std::make_unique<xrt::bo>(
                device, instructions_.size() * sizeof(uint32_t),
                XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
            std::memcpy(instruction_bo_->map<void *>(), instructions_.data(),
                        instructions_.size() * sizeof(uint32_t));
            instruction_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const size_t input_elements = static_cast<size_t>(kHeaderElements) +
                static_cast<size_t>(groups_) * kBatch * shape.k;
            input_bo_ = std::make_unique<xrt::bo>(
                device, input_elements * sizeof(uint16_t),
                XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
            weight_bo_ = std::make_unique<xrt::bo>(
                device, packed_.size(), XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(4));
            output_bo_ = std::make_unique<xrt::bo>(
                device, static_cast<size_t>(kBatch) * padded_n_ * sizeof(float),
                XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

            input_.resize(static_cast<size_t>(kBatch) * shape.k);
            auto * mapped = input_bo_->map<uint16_t *>();
            std::fill_n(mapped, input_elements, static_cast<uint16_t>(0));
            mapped[0] = float_to_bf16(
                static_cast<float>(shape.k / ember::xdna2::kQ8TileK));
            mapped[1] = float_to_bf16(static_cast<float>(groups_));
            for (int token = 0; token < kBatch; ++token) {
                for (int lane = 0; lane < shape.k; ++lane) {
                    const uint16_t bits = float_to_bf16(
                        static_cast<float>((lane + token) % 31 - 15) / 32.0f);
                    input_[static_cast<size_t>(token) * shape.k + lane] =
                        bf16_to_float(bits);
                    for (int group = 0; group < groups_; ++group) {
                        const size_t destination =
                            static_cast<size_t>(kHeaderElements) +
                            (static_cast<size_t>(group) * kBatch + token) *
                                shape.k +
                            lane;
                        mapped[destination] = bits;
                    }
                }
            }
            std::memcpy(weight_bo_->map<void *>(), packed_.data(), packed_.size());
            input_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weight_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        const Shape & shape() const { return shape_; }
        int padded_n() const { return padded_n_; }
        int groups() const { return groups_; }
        xrt::bo & instructions() { return *instruction_bo_; }
        uint32_t instruction_words() const {
            return static_cast<uint32_t>(instructions_.size());
        }
        xrt::bo & input_bo() { return *input_bo_; }
        xrt::bo & weight_bo() { return *weight_bo_; }
        xrt::bo & output_bo() { return *output_bo_; }

        Metrics validate() {
            output_bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const float * staging = output_bo_->map<const float *>();
            std::vector<float> expected(
                static_cast<size_t>(kBatch) * shape_.n);
            double error_sum = 0.0;
            double dot = 0.0;
            double actual_norm = 0.0;
            double expected_norm = 0.0;
            Metrics metrics;
            for (int token = 0; token < kBatch; ++token) {
                if (!ember::xdna2::q8_gemm_raw_reference(
                        projection_.raw.data(), projection_.raw.size(),
                        input_.data() + static_cast<size_t>(token) * shape_.k,
                        shape_.k, shape_.n,
                        expected.data() + static_cast<size_t>(token) * shape_.n))
                    throw std::runtime_error("Q8 reference failed");
                for (int output = 0; output < shape_.n; ++output) {
                    const int group = output / kOutputsPerGroup;
                    const int within_group = output % kOutputsPerGroup;
                    const int row = within_group /
                        ember::xdna2::kQ8OutputsPerRow;
                    const int within_row = within_group %
                        ember::xdna2::kQ8OutputsPerRow;
                    const int column = within_row / ember::xdna2::kQ8TileN;
                    const int lane = within_row % ember::xdna2::kQ8TileN;
                    const size_t source =
                        static_cast<size_t>(group) * kBatch * kOutputsPerGroup +
                        static_cast<size_t>(row) *
                            ember::xdna2::kQ8AieColumns * kBatch *
                            ember::xdna2::kQ8TileN +
                        static_cast<size_t>(column) * kBatch *
                            ember::xdna2::kQ8TileN +
                        static_cast<size_t>(token) * ember::xdna2::kQ8TileN +
                        lane;
                    const double actual = staging[source];
                    const double reference =
                        expected[static_cast<size_t>(token) * shape_.n + output];
                    const float error = static_cast<float>(
                        std::fabs(actual - reference));
                    metrics.max_abs = std::max(metrics.max_abs, error);
                    error_sum += error;
                    dot += actual * reference;
                    actual_norm += actual * actual;
                    expected_norm += reference * reference;
                }
            }
            const size_t count = static_cast<size_t>(kBatch) * shape_.n;
            metrics.mean_abs = error_sum / static_cast<double>(count);
            metrics.cosine = dot / std::sqrt(actual_norm * expected_norm);
            return metrics;
        }

    private:
        Shape shape_;
        int padded_n_;
        int groups_;
        ember::xdna2::Q8ModelProjection projection_;
        std::vector<uint8_t> packed_;
        std::vector<uint32_t> instructions_;
        std::vector<float> input_;
        std::unique_ptr<xrt::bo> instruction_bo_;
        std::unique_ptr<xrt::bo> input_bo_;
        std::unique_ptr<xrt::bo> weight_bo_;
        std::unique_ptr<xrt::bo> output_bo_;
    };

    UniversalOverlay(xrt::device & device, const std::string & image)
        : xclbin_(image) {
        const auto kernels = xclbin_.get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & candidate) {
                return candidate.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) throw std::runtime_error("AIE kernel absent");
        device.register_xclbin(xclbin_);
        context_ = std::make_unique<xrt::hw_context>(device, xclbin_.get_uuid());
        kernel_ = std::make_unique<xrt::kernel>(*context_, found->get_name());
        dummy6_ = std::make_unique<xrt::bo>(
            device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(6));
        dummy7_ = std::make_unique<xrt::bo>(
            device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel_->group_id(7));
    }

    std::unique_ptr<Workload> make_workload(
            xrt::device & device, const std::string & directory,
            const std::string & model, const Shape & shape) {
        return std::make_unique<Workload>(
            device, *kernel_, directory, model, shape);
    }

    void run(Workload & workload) {
        xrt::run command(*kernel_);
        command.set_arg(0, 3);
        command.set_arg(1, workload.instructions());
        command.set_arg(2, workload.instruction_words());
        command.set_arg(3, workload.input_bo());
        command.set_arg(4, workload.weight_bo());
        command.set_arg(5, workload.output_bo());
        command.set_arg(6, *dummy6_);
        command.set_arg(7, *dummy7_);
        command.start();
        if (command.wait() != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("universal projection command failed");
    }

private:
    xrt::xclbin xclbin_;
    std::unique_ptr<xrt::hw_context> context_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::unique_ptr<xrt::bo> dummy6_;
    std::unique_ptr<xrt::bo> dummy7_;
};

double time_repeated(UniversalOverlay & overlay,
                     UniversalOverlay::Workload & workload, int repeats) {
    const auto begin = Clock::now();
    for (int run = 0; run < repeats; ++run) overlay.run(workload);
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count() /
           static_cast<double>(repeats);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: %s ARTIFACT_DIR DRAFT_GGUF [REPEATS]\n",
                     argv[0]);
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
        const std::string directory = argv[1];
        const std::string model = argv[2];
        xrt::device device(0);
        UniversalOverlay overlay(
            device, directory + "/q8_projection_universal_v4_b5.xclbin");
        std::vector<std::unique_ptr<UniversalOverlay::Workload>> workloads;
        for (const Shape & shape : kShapes)
            workloads.push_back(
                overlay.make_workload(device, directory, model, shape));

        double standalone_total = 0.0;
        for (auto & workload : workloads) {
            overlay.run(*workload);
            const Metrics metrics = workload->validate();
            const double milliseconds =
                time_repeated(overlay, *workload, repeats);
            standalone_total += milliseconds;
            std::printf(
                "xdna_universal shape=%s K=%d N=%d padded_N=%d groups=%d "
                "sequence_ms=%.6f max_abs=%.8g mean_abs=%.8g cosine=%.10f\n",
                workload->shape().label, workload->shape().k,
                workload->shape().n, workload->padded_n(), workload->groups(),
                milliseconds, metrics.max_abs, metrics.mean_abs, metrics.cosine);
            if (metrics.cosine < 0.99999 || metrics.max_abs > 0.01f)
                throw std::runtime_error(std::string("accuracy gate failed for ") +
                                         workload->shape().label);
        }
        const auto cycle_begin = Clock::now();
        for (int run = 0; run < repeats; ++run)
            for (auto & workload : workloads) overlay.run(*workload);
        const double cycle_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - cycle_begin).count() /
            static_cast<double>(repeats);
        std::printf(
            "xdna_universal_cycle repeats=%d standalone_total_ms=%.6f "
            "cycle_ms=%.6f control_overhead_ms=%.6f\n",
            repeats, standalone_total, cycle_ms, cycle_ms - standalone_total);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "XDNA universal projection validation failed: %s\n",
                     exception.what());
        return 1;
    }
}
