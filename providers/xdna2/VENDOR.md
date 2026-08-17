# XDNA2 kernel provenance

The full-array object-FIFO topology in `kernel/rocmfp2_gemv.py` is adapted
from TileFuse's W4A16 vector-matrix example:

- repository: <https://github.com/glassescrab/mlir-aie>
- branch at import: `feature/update-mix-mm-int4-verification`
- commit: `8c3d2be63161c255c4e290e500702571c69a7112`
- upstream path:
  `programming_examples/ablation_study/vm_w4a16_with_all/vector_matrix/vector_matrix.py`
- license: Apache-2.0 with LLVM exception

Ember changed the packed-weight tile from asymmetric INT4 to the byte-exact
GGML ROCMFP2 affine block (`32 values + UE4M3 scale + UE4M3 offset`) and
simplified DMA around Ember's pre-tiled layout. Generations 1-3 use Ember's
scalar correctness microkernel. Generation 4 adapts TileFuse's vector
dequantize-then-accumulate structure from
`aie_kernels/aie2/vm_mix_int4_64x8.cc`: its cache-time pack expands FP2 codes
to uint4 nibbles, stores exact BF16 affine metadata, then uses native AIE2P
unpack/dequantize/vector-MAC instructions. TileFuse's measured INT4 performance
is not claimed for Ember; Gen4 is measured independently on gfx1151.

The runtime image's XRT base and XDNA shim are built from AMD's driver tree:

- repository: <https://github.com/amd/xdna-driver>
- commit: `455fc6be78e9cdf8b41a1547eff0e30351f21fec`
- XRT submodule: `8b60ae7a90bfbc873e181497fd34ca520b4ef504`
- license: Apache-2.0

Only userspace XRT and the XDNA shim belong in the container. The kernel driver
and firmware belong on the host and `/dev/accel/accel0` is passed through.

Generation 5's fused-expert investigation also consulted, without copying
source from, AMD's current IRON operator and sequencing examples:

- repository: <https://github.com/amd/IRON>
- commit: `cdc48e93fd2c8776105780790c46ba4bca1bc40e`
- relevant paths: `iron/operators/swiglu_decode/`, `iron/operators/gemm/`,
  `iron/operators/mha/`, `iron/operators/rms_norm/`,
  `iron/common/compilation/sequence.py`, and `aie_kernels/aie2p/silu.cc`
- license: Apache-2.0

The low-level DMA/task-queue contract was checked against AIE-RT branch
`release/main_aig` at commit `8849e208bdcc533b20a0ed3f95c1ce961dee9c3a`.
In particular, `driver/src/dma/xaie_dma.c` documents
`XAie_DmaChannelSetStartQueue(..., EnTokenIssue)` as issuing a token when the
queued task completes, while `driver/src/global/xaie2ipugbl_reginit.c` marks
that start-queue field available on AIE2IPU. These are research references,
not vendored dependencies. Ember's activation contract and accuracy thresholds
remain independent.

MLIR-AIE itself vendors that AIE-RT branch as a **compiler-side** register and
transaction source, with local fixes for backend selection and ELF `.bss`
zero-fill.  Ember therefore keeps AIE-RT out of the runtime image: `aiecc.py`
serializes configuration into the packaged `.insts`, and XRT/amdnpu executes
that immutable stream.  This boundary is also a compatibility safeguard.  The
current MLIR-AIE source explicitly warns that its NPU firmware transaction
opcodes are newer than the enum in the pinned AIE-RT source; dynamically
building a second transaction stream from `libxaiengine` would risk mixing the
two layouts.  Kernel artifacts, compiler wheel, XRT userspace, and host driver
must be validated as one tuple instead.

Generations 7 and 8 also use only the public AIE API operations demonstrated by the
official MLIR-AIE tree (uint4 unpack, vector masks/select, BF16 conversion and
whole-array object FIFOs). The current upstream tree was rechecked at commit
`c495b2b4f988d81043b5a3cbea3be223c1c7a93c`, including the BF16 norm, RoPE,
softmax, matrix-multiplication, and activation-epilogue examples. Ember's
ROCMFP4 signed-codebook mapping, five-row expert graphs, route mask, packer, and
numerical contract are original to this prototype. The AIE-RT queue findings
above are why both generations retain explicit completion-producing tasks
rather than assuming that queued descriptor reuse is safe. Gen8's resident
expert selection uses XRT's public `bo(parent, size, offset)` sub-buffer API;
it does not call AIE-RT directly or vendor AIE-RT source.

Generation 6's weight-reusing batch topology also consulted, without copying
source from, the official MLIR-AIE whole-array matrix examples:

- repository: <https://github.com/Xilinx/mlir-aie>
- commit inspected: `c95544269f0c074d6d3e213ee43cc34dc4100801`
- relevant path:
  `programming_examples/ml/block_datatypes/matrix_multiplication/whole_array_mixed/`
- license: Apache-2.0 with LLVM exception

The transferable design is to distribute output ownership, broadcast reused
inputs, and keep partial sums local. Ember's ROCMFP2 decode, four-row shared
expert shape, packet replay, and accuracy contract are independent. A native
AIE2P BF16 matrix-instruction experiment derived from the public AIE API was
hardware-rejected after timing out and is not present in the shipped Gen6
microkernel.

The host/kernel optimization audit also consulted, without copying source
from, the public llama.cpp XDNA2 backend and its corrected performance record:

- repository: <https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU>
- commit inspected: `9bfe13a0a78415806e81d22748478aeade0b4bd6`
- relevant paths: `ggml/src/ggml-xdna/` and `docs/xdna-npu/phase12.html`
- license: MIT

Its useful transferable result is the measured importance of balanced AIE
microtiles and instruction-stream length. Its earlier NPU-decode headline was
later corrected by the project itself as accidental Vulkan execution, so
Ember does not use that number as evidence of NPU decode performance.
