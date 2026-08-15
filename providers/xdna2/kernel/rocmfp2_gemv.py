# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Full-array XDNA2 streaming GEMV, adapted from TileFuse's vector_matrix.py at
# mlir-aie commit 8c3d2be63161c255c4e290e500702571c69a7112.

import argparse
import numpy as np
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import (
    AIEDevice,
    ObjectFifoPort,
    buffer,
    core,
    device,
    external_func,
    object_fifo,
    object_fifo_link,
    tile,
)
from aie.dialects.aiex import dma_wait, npu_dma_memcpy_nd, runtime_sequence
from aie.helpers.dialects.scf import _for as range_


TILE_K = 128
TILE_N = 64
AIE_ROWS = 4
AIE_COLS = 8
ROCMFP2_TILE_BYTES = TILE_N * (TILE_K // 32) * 10
ARRAY_OUTPUTS = TILE_N * AIE_ROWS * AIE_COLS


def build_gemv(k_total, n_total, generation):
    if k_total <= 0 or k_total % TILE_K:
        raise ValueError("K must be a positive multiple of 128")
    if n_total <= 0 or n_total % ARRAY_OUTPUTS:
        raise ValueError("N must be a positive multiple of 2048")

    k_tiles = k_total // TILE_K
    output_groups = n_total // ARRAY_OUTPUTS
    packed_bytes = k_total * n_total // 32 * 10

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            input_tile_ty = np.ndarray[(TILE_K,), np.dtype[bfloat16]]
            weight_tile_ty = np.ndarray[(ROCMFP2_TILE_BYTES,), np.dtype[np.uint8]]
            weight_mem_ty = np.ndarray[
                (ROCMFP2_TILE_BYTES * AIE_ROWS,), np.dtype[np.uint8]
            ]
            output_tile_ty = np.ndarray[(TILE_N,), np.dtype[bfloat16]]
            output_mem_ty = np.ndarray[
                (TILE_N * AIE_ROWS,), np.dtype[bfloat16]
            ]
            accumulator_ty = np.ndarray[(TILE_N,), np.dtype[np.float32]]
            if generation == 1:
                zero_name = "zero_rocmfp2_bf16"
                gemv_name = "gemv_rocmfp2_bf16"
            else:
                zero_name = "zero_rocmfp2_f32"
                gemv_name = "gemv_rocmfp2_f32"

            zero = external_func(
                zero_name,
                inputs=[output_tile_ty if generation == 1 else accumulator_ty],
                link_with="rocmfp2_gemv.o",
            )
            gemv = external_func(
                gemv_name,
                inputs=[input_tile_ty, weight_tile_ty,
                        output_tile_ty if generation == 1 else accumulator_ty],
                link_with="rocmfp2_gemv.o",
            )
            if generation == 2:
                store = external_func(
                    "store_rocmfp2_f32_bf16",
                    inputs=[accumulator_ty, output_tile_ty],
                    link_with="rocmfp2_gemv.o",
                )

            shims = [tile(col, 0) for col in range(AIE_COLS)]
            mems = [tile(col, 1) for col in range(AIE_COLS)]
            cores = [
                [tile(col, row + 2) for col in range(AIE_COLS)]
                for row in range(AIE_ROWS)
            ]
            accumulators = None
            if generation == 2:
                accumulators = [
                    [buffer(cores[row][col], accumulator_ty,
                            name=f"accumulator_{row}_{col}")
                     for col in range(AIE_COLS)]
                    for row in range(AIE_ROWS)
                ]

            mem_a = [None] * AIE_COLS
            in_a = [None] * AIE_COLS
            mem_b = [None] * AIE_COLS
            in_b = [[None] * AIE_COLS for _ in range(AIE_ROWS)]
            mem_c = [None] * AIE_COLS
            out_c = [[None] * AIE_COLS for _ in range(AIE_ROWS)]

            for col in range(AIE_COLS):
                mem_a[col] = object_fifo(
                    f"mem_A{col}", shims[col], mems[col], 2, input_tile_ty
                )
                in_a[col] = object_fifo(
                    f"in_A{col}", mems[col],
                    [cores[row][col] for row in range(AIE_ROWS)],
                    2, input_tile_ty,
                )
                object_fifo_link(mem_a[col], in_a[col])

                mem_b[col] = object_fifo(
                    f"mem_B{col}", shims[col], mems[col], 2, weight_mem_ty
                )
                for row in range(AIE_ROWS):
                    in_b[row][col] = object_fifo(
                        f"in_B{row}_{col}", mems[col], cores[row][col],
                        2, weight_tile_ty,
                    )
                object_fifo_link(
                    mem_b[col],
                    [in_b[row][col] for row in range(AIE_ROWS)],
                    [],
                    [row * ROCMFP2_TILE_BYTES for row in range(AIE_ROWS)],
                )

                mem_c[col] = object_fifo(
                    f"mem_C{col}", mems[col], shims[col], 2, output_mem_ty
                )
                for row in range(AIE_ROWS):
                    out_c[row][col] = object_fifo(
                        f"out_C{row}_{col}", cores[row][col], mems[col],
                        2, output_tile_ty,
                    )
                object_fifo_link(
                    [out_c[row][col] for row in range(AIE_ROWS)],
                    mem_c[col],
                    [row * TILE_N for row in range(AIE_ROWS)],
                )

            for row in range(AIE_ROWS):
                for col in range(AIE_COLS):
                    accumulator = (
                        accumulators[row][col] if generation == 2 else None
                    )

                    @core(cores[row][col])
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            output = out_c[row][col].acquire(
                                ObjectFifoPort.Produce, 1
                            )
                            partial = output if generation == 1 else accumulator
                            zero(partial)
                            for _ in range_(k_tiles):
                                activation = in_a[col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                weight = in_b[row][col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                gemv(activation, weight, partial)
                                in_a[col].release(ObjectFifoPort.Consume, 1)
                                in_b[row][col].release(ObjectFifoPort.Consume, 1)
                            if generation == 2:
                                store(partial, output)
                            out_c[row][col].release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(k_total,), np.dtype[bfloat16]],
                np.ndarray[(packed_bytes,), np.dtype[np.uint8]],
                np.ndarray[(n_total,), np.dtype[bfloat16]],
            )
            def sequence(activation, weights, output):
                bytes_per_column = k_tiles * AIE_ROWS * ROCMFP2_TILE_BYTES
                for group in range(output_groups):
                    for col in range(AIE_COLS):
                        npu_dma_memcpy_nd(
                            metadata=mem_a[col], bd_id=2, mem=activation,
                            sizes=[1, 1, 1, k_total],
                            strides=[0, 0, 0, 1],
                        )
                        weight_offset = (
                            group * AIE_COLS * bytes_per_column
                            + col * bytes_per_column
                        )
                        npu_dma_memcpy_nd(
                            metadata=mem_b[col], bd_id=1, mem=weights,
                            offsets=[0, 0, 0, weight_offset],
                            sizes=[1, 1, 1, bytes_per_column],
                            strides=[0, 0, 0, 1],
                        )
                        output_offset = group * ARRAY_OUTPUTS + col * TILE_N
                        npu_dma_memcpy_nd(
                            metadata=mem_c[col], bd_id=0, mem=output,
                            offsets=[0, 0, 0, output_offset],
                            sizes=[1, 1, AIE_ROWS, TILE_N],
                            strides=[0, 0, AIE_COLS * TILE_N, 1],
                        )
                    dma_wait(*mem_c)

        print(ctx.module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("--generation", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    build_gemv(args.K, args.N, args.generation)
