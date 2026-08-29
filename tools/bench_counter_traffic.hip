// Deterministic global-memory traffic for ROCm counter-unit calibration.
//
// Each active thread owns one 64-byte cache line.  The read, write, and copy
// kernels therefore have an exact logical traffic count of
// lines * 64 * repetitions.  Buffers used by the harness are larger than the
// gfx1151 MALL so the read stream cannot be satisfied from the last-level
// cache.  This is a calibration workload, not an Ember performance benchmark.
//
// Build inside the pinned ROCm image:
//   hipcc -O3 --offload-arch=gfx1151 -o bench_counter_traffic \
//     tools/bench_counter_traffic.hip

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr std::size_t kLineBytes = 64;
constexpr int kThreads = 256;

#define HIP_CHECK(call) do { \
    const hipError_t error__ = (call); \
    if (error__ != hipSuccess) { \
      std::fprintf(stderr, "HIP failure at %s:%d: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(error__)); \
      return 2; \
    } \
  } while (0)

__global__ void read_lines(const std::uint4 *source, std::size_t lines) {
  const std::size_t line = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
  if (line >= lines) return;
  const volatile std::uint4 *p = source + line * 4;
  const std::uint4 a = p[0];
  const std::uint4 b = p[1];
  const std::uint4 c = p[2];
  const std::uint4 d = p[3];
  // Keep the loads observable without adding a material output stream.
  asm volatile("" : : "r"(a.x), "r"(b.y), "r"(c.z), "r"(d.w));
}

__global__ void write_lines(std::uint4 *destination, std::size_t lines) {
  const std::size_t line = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
  if (line >= lines) return;
  volatile std::uint4 *p = destination + line * 4;
  const std::uint4 value = make_uint4(0x13579bdfU, 0x2468ace0U,
                                      0x0f1e2d3cU, 0x4b5a6978U);
  p[0] = value;
  p[1] = value;
  p[2] = value;
  p[3] = value;
}

__global__ void copy_lines(const std::uint4 *source, std::uint4 *destination,
                           std::size_t lines) {
  const std::size_t line = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
  if (line >= lines) return;
  const volatile std::uint4 *s = source + line * 4;
  volatile std::uint4 *d = destination + line * 4;
  d[0] = s[0];
  d[1] = s[1];
  d[2] = s[2];
  d[3] = s[3];
}

void usage(const char *program) {
  std::fprintf(stderr,
      "usage: %s --mode read|write|copy|noop --bytes N --reps N [--warmup N]\n",
      program);
}

}  // namespace

int main(int argc, char **argv) {
  const char *mode = nullptr;
  std::size_t bytes = 0;
  int reps = 0;
  int warmup = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
    else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc)
      bytes = std::strtoull(argv[++i], nullptr, 10);
    else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
      reps = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
    else { usage(argv[0]); return 2; }
  }
  if (mode == nullptr || bytes < kLineBytes || bytes % kLineBytes != 0 ||
      reps < 1 || warmup < 0 ||
      (std::strcmp(mode, "read") != 0 && std::strcmp(mode, "write") != 0 &&
       std::strcmp(mode, "copy") != 0 && std::strcmp(mode, "noop") != 0)) {
    usage(argv[0]); return 2;
  }

  const std::size_t lines = bytes / kLineBytes;
  const std::size_t allocation = lines * 4 * sizeof(std::uint4);
  std::uint4 *source = nullptr;
  std::uint4 *destination = nullptr;
  HIP_CHECK(hipMalloc(&source, allocation));
  if (std::strcmp(mode, "write") != 0 && std::strcmp(mode, "noop") != 0)
    HIP_CHECK(hipMalloc(&destination, allocation));
  if (source != nullptr) HIP_CHECK(hipMemset(source, 0, allocation));
  if (destination != nullptr) HIP_CHECK(hipMemset(destination, 0, allocation));

  const dim3 block(kThreads);
  const dim3 grid(static_cast<unsigned>((lines + kThreads - 1) / kThreads));
  auto launch = [&]() {
    if (std::strcmp(mode, "read") == 0) read_lines<<<grid, block>>>(source, lines);
    else if (std::strcmp(mode, "write") == 0) write_lines<<<grid, block>>>(source, lines);
    else if (std::strcmp(mode, "copy") == 0) copy_lines<<<grid, block>>>(source, destination, lines);
    else hipLaunchKernelGGL(read_lines, grid, block, 0, 0, source, std::size_t{0});
  };
  for (int i = 0; i < warmup; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < reps; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());

  const std::uint64_t traffic = static_cast<std::uint64_t>(bytes) * reps;
  std::printf("{\"mode\":\"%s\",\"bytes_per_rep\":%zu,\"repetitions\":%d,"
              "\"expected_read_bytes\":%llu,\"expected_write_bytes\":%llu}\n",
              mode, bytes, reps,
              static_cast<unsigned long long>(
                  (std::strcmp(mode, "write") == 0 || std::strcmp(mode, "noop") == 0)
                      ? 0 : traffic),
              static_cast<unsigned long long>(
                  (std::strcmp(mode, "read") == 0 || std::strcmp(mode, "noop") == 0)
                      ? 0 : traffic));
  if (destination != nullptr) HIP_CHECK(hipFree(destination));
  HIP_CHECK(hipFree(source));
  return 0;
}
