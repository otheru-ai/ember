# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""128-row compensated-Q8 GEMM for DSpark context projections.

The topology follows AMD IRON's full-array GEMM: four compute rows partition
M into 32-token tiles and eight columns partition N into 64-wide tiles. Each
activation tile is multicast across a compute row and each corrected weight
tile across a compute column, so 128 tokens reuse each streamed weight.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import (
    AIEDevice, ObjectFifoPort, core, device, external_func, object_fifo,
    object_fifo_link, tile,
)
from aie.dialects.aiex import dma_wait, npu_dma_memcpy_nd, runtime_sequence
from aie.helpers.dialects.scf import _for as range_


BATCH = 128
M_TILE = 32
TILE_K = 128
TILE_N = 64
ROWS = 4
COLS = 8
CORRECTED_TILE_BF16 = 2 * TILE_K * TILE_N
OUTPUT_TILE_F32 = M_TILE * TILE_N


def build_gemm(k, n, generation=4):
    if k <= 0 or k % 4096 or n <= 0 or n % (COLS * TILE_N):
        raise ValueError("K must be a multiple of 4096; N of 512")
    k_tiles = k // TILE_K
    descriptor_k_tiles = 32
    k_tile_blocks = k_tiles // descriptor_k_tiles
    groups = n // (COLS * TILE_N)
    if generation not in (4, 5):
        raise ValueError("generation must be 4 or 5")
    suffix = "v5" if generation == 5 else "v4"
    object_name = (
        "q8_gemm_m32_v5.o" if generation == 5 else "q8_gemm_m32_v4.o"
    )
    # Gap every 32-KiB corrected tile. Without this sentinel, canonicalization
    # coalesces four adjacent objects into a 128-KiB shim transfer, beyond the
    # 64-KiB contiguous descriptor field used by this graph.
    tile_stride_bf16 = CORRECTED_TILE_BF16 + 2
    packed_bf16 = COLS * groups * k_tiles * tile_stride_bf16

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            activation_ty = np.ndarray[
                (M_TILE * TILE_K,), np.dtype[bfloat16]
            ]
            weight_ty = np.ndarray[
                (CORRECTED_TILE_BF16,), np.dtype[bfloat16]
            ]
            output_ty = np.ndarray[
                (OUTPUT_TILE_F32,), np.dtype[np.float32]
            ]
            output_mem_ty = np.ndarray[
                (ROWS * OUTPUT_TILE_F32,), np.dtype[np.float32]
            ]

            zero = external_func(
                f"zero_q8_{suffix}_f32_m32", inputs=[output_ty],
                link_with=object_name,
            )
            gemm = external_func(
                f"gemm_q8_{suffix}_f32_m32",
                inputs=[activation_ty, weight_ty, output_ty],
                link_with=object_name,
            )

            shims = [tile(col, 0) for col in range(COLS)]
            mems = [tile(col, 1) for col in range(COLS)]
            cores = [
                [tile(col, row + 2) for col in range(COLS)]
                for row in range(ROWS)
            ]

            in_a = [None] * ROWS
            in_w = [None] * COLS
            mem_o = [None] * COLS
            out_o = [[None] * COLS for _ in range(ROWS)]
            for row in range(ROWS):
                # Even shim columns give each activation multicast its own
                # physical source while preserving one source per AIE row.
                in_a[row] = object_fifo(
                    f"q8{suffix}_A_{row}", shims[2 * row], cores[row], 1,
                    activation_ty,
                )
            for col in range(COLS):
                destinations = [cores[row][col] for row in range(ROWS)]
                in_w[col] = object_fifo(
                    f"q8{suffix}_W_{col}", shims[col], destinations, 1,
                    weight_ty,
                )
                mem_o[col] = object_fifo(
                    f"q8{suffix}_mem_O_{col}", mems[col], shims[col], 1,
                    output_mem_ty,
                )
                for row in range(ROWS):
                    out_o[row][col] = object_fifo(
                        f"q8{suffix}_O_{row}_{col}", cores[row][col],
                        mems[col],
                        1, output_ty,
                    )
                object_fifo_link(
                    [out_o[row][col] for row in range(ROWS)], mem_o[col],
                    [row * OUTPUT_TILE_F32 for row in range(ROWS)],
                )

            for row in range(ROWS):
                for col in range(COLS):
                    @core(cores[row][col])
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            for _ in range_(groups):
                                result = out_o[row][col].acquire(
                                    ObjectFifoPort.Produce, 1
                                )
                                zero(result)
                                for _ in range_(k_tiles):
                                    activation = in_a[row].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    weight = in_w[col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    gemm(activation, weight, result)
                                    in_w[col].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    in_a[row].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                out_o[row][col].release(
                                    ObjectFifoPort.Produce, 1
                                )

            host_input_ty = np.ndarray[
                (groups * BATCH * k,), np.dtype[bfloat16]
            ]
            host_weight_ty = np.ndarray[
                (packed_bf16,), np.dtype[bfloat16]
            ]
            host_output_ty = np.ndarray[
                (BATCH * n,), np.dtype[np.float32]
            ]

            @runtime_sequence(host_input_ty, host_weight_ty, host_output_ty)
            def sequence(host_input, weights, output):
                for col in range(COLS):
                    npu_dma_memcpy_nd(
                        metadata=mem_o[col], bd_id=0, mem=output,
                        offsets=[0, 0, 0, col * OUTPUT_TILE_F32],
                        sizes=[groups, ROWS, 4, OUTPUT_TILE_F32 // 4],
                        strides=[
                            ROWS * COLS * OUTPUT_TILE_F32,
                            COLS * OUTPUT_TILE_F32, OUTPUT_TILE_F32 // 4, 1,
                        ],
                    )
                for row in range(ROWS):
                    npu_dma_memcpy_nd(
                        metadata=in_a[row], bd_id=4, mem=host_input,
                        offsets=[
                            0, 0, 0,
                            row * groups * k_tiles * M_TILE * TILE_K,
                        ],
                        sizes=[
                            groups * k_tile_blocks, descriptor_k_tiles, 8,
                            M_TILE * TILE_K // 8,
                        ],
                        strides=[
                            descriptor_k_tiles * M_TILE * TILE_K,
                            M_TILE * TILE_K, M_TILE * TILE_K // 8, 1,
                        ],
                    )
                for col in range(COLS):
                    npu_dma_memcpy_nd(
                        metadata=in_w[col], bd_id=5, mem=weights,
                        offsets=[
                            0, 0, 0,
                            col * groups * k_tiles * tile_stride_bf16,
                        ],
                        sizes=[
                            groups * k_tile_blocks, descriptor_k_tiles, 32,
                            CORRECTED_TILE_BF16 // 32,
                        ],
                        strides=[
                            descriptor_k_tiles * tile_stride_bf16,
                            tile_stride_bf16,
                            CORRECTED_TILE_BF16 // 32, 1,
                        ],
                    )
                dma_wait(*mem_o)

        print(ctx.module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("--generation", type=int, choices=(4, 5), default=4)
    args = parser.parse_args()
    build_gemm(args.K, args.N, args.generation)
