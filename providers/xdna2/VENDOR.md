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
- relevant paths: `iron/operators/swiglu_decode/`,
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
