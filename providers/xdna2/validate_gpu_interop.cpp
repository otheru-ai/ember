// Physical gfx1151/XDNA2 dma-buf interoperability probe.
//
// This is deliberately separate from the provider ABI.  It proves that an
// allocation owned by ROCr can be attached to amdxdna through DRM PRIME and
// used as both an AIE input and output without a CPU staging allocation.  The
// explicit HIP completion and XRT BO syncs are the device hand-off contract;
// unified physical memory alone does not imply cross-driver ordering.

#include "rocmfp2_pack.h"

#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInput = 4096;
constexpr int kOutput = 2048;

void check_hip(hipError_t status, const char * operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 hipGetErrorString(status));
    }
}

void check_hsa(hsa_status_t status, const char * operation) {
    if (status == HSA_STATUS_SUCCESS) return;
    const char * detail = nullptr;
    hsa_status_string(status, &detail);
    throw std::runtime_error(std::string(operation) + ": " +
                             (detail ? detail : "unknown HSA error"));
}

std::vector<uint32_t> read_instructions(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path);
    const std::streamoff length = file.tellg();
    if (length <= 0 || length % static_cast<std::streamoff>(sizeof(uint32_t)))
        throw std::runtime_error("invalid instruction stream " + path);
    file.seekg(0);
    std::vector<uint32_t> words(
        static_cast<size_t>(length) / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char *>(words.data()), length))
        throw std::runtime_error("short instruction stream " + path);
    return words;
}

void fill_identity_rows(std::vector<uint8_t> & raw) {
    const int blocks_per_row = kInput / ember::xdna2::kRocmfp2BlockWeights;
    std::fill(raw.begin(), raw.end(), 0);
    for (int output = 0; output < kOutput; ++output) {
        const int input = output % kInput;
        const int block = input / ember::xdna2::kRocmfp2BlockWeights;
        const int lane = input % ember::xdna2::kRocmfp2BlockWeights;
        uint8_t * q = raw.data() +
            (static_cast<size_t>(output) * static_cast<size_t>(blocks_per_row) +
             static_cast<size_t>(block)) * ember::xdna2::kRocmfp2BlockBytes;
        q[lane >> 2] = static_cast<uint8_t>(1u << (2 * (lane & 3)));
        q[8] = 0x40;  // UE4M3 1.0; offset remains zero.
    }
}

class HipAllocation {
public:
    HipAllocation(size_t bytes, bool managed) : bytes_(bytes) {
        const hipError_t status = managed
            ? hipMallocManaged(&pointer_, bytes_, hipMemAttachGlobal)
            : hipMalloc(&pointer_, bytes_);
        check_hip(status, managed ? "hipMallocManaged" : "hipMalloc");
    }

    ~HipAllocation() {
        if (pointer_) (void) hipFree(pointer_);
    }

    HipAllocation(const HipAllocation &) = delete;
    HipAllocation & operator=(const HipAllocation &) = delete;

    void * data() const { return pointer_; }
    size_t size() const { return bytes_; }

private:
    void * pointer_ = nullptr;
    size_t bytes_ = 0;
};

class ImportedBo {
public:
    ImportedBo(xrt::device & device, void * pointer, size_t bytes) {
        int fd = -1;
        uint64_t offset = 0;
        check_hsa(hsa_amd_portable_export_dmabuf(pointer, bytes, &fd, &offset),
                  "hsa_amd_portable_export_dmabuf");
        try {
            parent_ = std::make_unique<xrt::bo>(
                device, static_cast<xrt::bo::export_handle>(fd));
            if (offset != 0 || parent_->size() != bytes) {
                if (offset + bytes > parent_->size())
                    throw std::runtime_error("exported dma-buf range is truncated");
                view_ = std::make_unique<xrt::bo>(*parent_, bytes,
                                                  static_cast<size_t>(offset));
            }
            offset_ = offset;
        } catch (...) {
            hsa_amd_portable_close_dmabuf(fd);
            throw;
        }
        check_hsa(hsa_amd_portable_close_dmabuf(fd),
                  "hsa_amd_portable_close_dmabuf");
    }

    xrt::bo & get() { return view_ ? *view_ : *parent_; }
    uint64_t offset() const { return offset_; }

private:
    std::unique_ptr<xrt::bo> parent_;
    std::unique_ptr<xrt::bo> view_;
    uint64_t offset_ = 0;
};

bool run_probe(xrt::device & device, xrt::kernel & kernel,
               xrt::bo & instruction_bo, uint32_t instruction_count,
               xrt::bo & weight_bo, xrt::bo & scratch_bo,
               xrt::bo & trace_bo, bool managed) {
    const size_t input_bytes = static_cast<size_t>(kInput) * sizeof(uint16_t);
    const size_t output_bytes = static_cast<size_t>(kOutput) * sizeof(float);
    HipAllocation hip_input(input_bytes, managed);
    HipAllocation hip_output(output_bytes, managed);

    std::vector<uint16_t> input(static_cast<size_t>(kInput), 0x3f80u);
    std::vector<float> poison(static_cast<size_t>(kOutput),
                              std::numeric_limits<float>::quiet_NaN());
    check_hip(hipMemcpy(hip_input.data(), input.data(), input_bytes,
                        hipMemcpyHostToDevice), "hipMemcpy input");
    check_hip(hipMemcpy(hip_output.data(), poison.data(), output_bytes,
                        hipMemcpyHostToDevice), "hipMemcpy output poison");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize before XDNA");

    const auto import_start = std::chrono::steady_clock::now();
    ImportedBo input_bo(device, hip_input.data(), input_bytes);
    ImportedBo output_bo(device, hip_output.data(), output_bytes);
    input_bo.get().sync(XCL_BO_SYNC_BO_TO_DEVICE);
    output_bo.get().sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto import_done = std::chrono::steady_clock::now();

    xrt::run run(kernel);
    run.set_arg(0, 3);
    run.set_arg(1, instruction_bo);
    run.set_arg(2, instruction_count);
    run.set_arg(3, input_bo.get());
    run.set_arg(4, weight_bo);
    run.set_arg(5, output_bo.get());
    run.set_arg(6, scratch_bo);
    run.set_arg(7, trace_bo);
    auto execute = [&]() {
        run.start();
        const auto state = run.wait();
        if (state != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("XDNA command did not complete");
        output_bo.get().sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    };
    execute();  // Exclude first-use context/kernel cost from the warm mean.
    constexpr int kTimedRuns = 20;
    const auto run_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < kTimedRuns; ++iteration) execute();
    const auto run_done = std::chrono::steady_clock::now();

    std::vector<float> output(static_cast<size_t>(kOutput));
    check_hip(hipMemcpy(output.data(), hip_output.data(), output_bytes,
                        hipMemcpyDeviceToHost), "hipMemcpy output");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize after XDNA");

    float max_abs = 0.0f;
    bool finite = true;
    for (float value : output) {
        finite = finite && std::isfinite(value);
        max_abs = std::max(max_abs, std::fabs(value - 1.0f));
    }
    const double import_ms = std::chrono::duration<double, std::milli>(
        import_done - import_start).count();
    const double warm_ms = std::chrono::duration<double, std::milli>(
        run_done - run_start).count() / kTimedRuns;
    std::printf("mode=%s input_offset=%llu output_offset=%llu "
                "input_addr=0x%llx output_addr=0x%llx import_ms=%.3f "
                "warm_run_sync_ms=%.3f max_abs=%.9g %s\n",
                managed ? "managed" : "device",
                static_cast<unsigned long long>(input_bo.offset()),
                static_cast<unsigned long long>(output_bo.offset()),
                static_cast<unsigned long long>(input_bo.get().address()),
                static_cast<unsigned long long>(output_bo.get().address()),
                import_ms, warm_ms, max_abs,
                finite && max_abs <= 1.0e-6f ? "PASS" : "FAIL");
    return finite && max_abs <= 1.0e-6f;
}

}  // namespace

int main() {
    try {
        check_hip(hipSetDevice(0), "hipSetDevice");
        xrt::device device(0);
        const char * configured = std::getenv("EMBER_XDNA_ARTIFACT_DIR");
        const std::string directory = configured && configured[0]
            ? configured : "/usr/local/share/ember/xdna2";
        const std::string base = directory + "/gemv_v4_4096x2048";
        xrt::xclbin xclbin(base + ".xclbin");
        const auto kernels = xclbin.get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel & candidate) {
                return candidate.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end())
            throw std::runtime_error("MLIR_AIE kernel not found");
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        xrt::kernel kernel(context, found->get_name());

        const std::vector<uint32_t> instructions =
            read_instructions(base + ".insts");
        xrt::bo instruction_bo(device,
            instructions.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
            kernel.group_id(1));
        std::memcpy(instruction_bo.map<void *>(), instructions.data(),
                    instructions.size() * sizeof(uint32_t));
        instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        std::vector<uint8_t> raw(
            ember::xdna2::rocmfp2_projection_bytes(kInput, kOutput));
        fill_identity_rows(raw);
        std::vector<uint8_t> packed;
        std::string pack_error;
        if (!ember::xdna2::pack_rocmfp2_gemv_v4(
                raw.data(), raw.size(), kInput, kOutput, packed, &pack_error))
            throw std::runtime_error("weight pack failed: " + pack_error);
        xrt::bo weight_bo(device, packed.size(), XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(4));
        std::memcpy(weight_bo.map<void *>(), packed.data(), packed.size());
        weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        xrt::bo scratch_bo(device, 1, XRT_BO_FLAGS_HOST_ONLY,
                           kernel.group_id(6));
        xrt::bo trace_bo(device, 1, XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(7));

        const uint32_t instruction_count =
            static_cast<uint32_t>(instructions.size());
        const bool device_ok = run_probe(
            device, kernel, instruction_bo, instruction_count,
            weight_bo, scratch_bo, trace_bo, false);
        bool managed_ok = true;
        if (std::getenv("EMBER_XDNA_INTEROP_TEST_MANAGED")) {
            managed_ok = run_probe(
                device, kernel, instruction_bo, instruction_count,
                weight_bo, scratch_bo, trace_bo, true);
        } else {
            std::printf("mode=managed SKIP "
                        "(set EMBER_XDNA_INTEROP_TEST_MANAGED=1 to probe)\n");
        }
        std::printf("GPU_XDNA_DMABUF_INTEROP_%s\n",
                    device_ok && managed_ok ? "PASS" : "FAIL");
        return device_ok && managed_ok ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GPU_XDNA_DMABUF_INTEROP_ERROR: %s\n",
                     error.what());
        return 1;
    }
}
