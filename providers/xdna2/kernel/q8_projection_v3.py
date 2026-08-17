# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Five-row compensated Q8 projection for real DSpark matrix shapes.

The runtime sequence deliberately describes one strided weight task per shim
column.  A memory tile fans each K tile out to the active compute rows, matching
the AIE-RT task/replay model while leaving host execution to XRT.
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


BATCH = 5
TILE_K = 128
TILE_N = 64
COLS = 8
MAX_ROWS = 4
CORRECTED_TILE_BF16 = 2 * TILE_K * TILE_N


def projection_rows(n):
    if n <= 0 or n % (COLS * TILE_N):
        return 0
    if n < MAX_ROWS * COLS * TILE_N:
        return n // (COLS * TILE_N)
    return MAX_ROWS if n % (MAX_ROWS * COLS * TILE_N) == 0 else 0


def build_projection(k, n):
    rows = projection_rows(n)
    if k <= 0 or k % TILE_K or not rows:
        raise ValueError(
            "K must be a multiple of 128; N must be 512/1024/1536 "
            "or a multiple of 2048"
        )
    k_tiles = k // TILE_K
    outputs_per_group = rows * COLS * TILE_N
    groups = n // outputs_per_group
    task_k_tiles = 4
    if k_tiles % task_k_tiles:
        raise ValueError("K must be a multiple of 512 for DMA task blocking")
    tasks_per_group = k_tiles // task_k_tiles
    task_payload_bf16 = task_k_tiles * rows * CORRECTED_TILE_BF16
    # Shim strides are byte-addressed and must be 32-bit aligned. Two BF16
    # sentinels also prevent the compiler from coalescing adjacent tasks into
    # a transfer larger than the AIE-RT one-MiB descriptor ceiling.
    task_stride_bf16 = task_payload_bf16 + 2
    packed_bf16 = COLS * groups * tasks_per_group * task_stride_bf16

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            activation_ty = np.ndarray[
                (BATCH * TILE_K,), np.dtype[bfloat16]
            ]
            weight_ty = np.ndarray[
                (CORRECTED_TILE_BF16,), np.dtype[bfloat16]
            ]
            weight_mem_ty = np.ndarray[
                (rows * CORRECTED_TILE_BF16,), np.dtype[bfloat16]
            ]
            output_ty = np.ndarray[(BATCH * TILE_N,), np.dtype[np.float32]]
            output_mem_ty = np.ndarray[
                (rows * BATCH * TILE_N,), np.dtype[np.float32]
            ]

            zero = external_func(
                "zero_q8_v2_f32x5", inputs=[output_ty],
                link_with="q8_gemm_v2.o"
            )
            gemm = external_func(
                "gemm_q8_v2_f32x5",
                inputs=[activation_ty, weight_ty, output_ty],
                link_with="q8_gemm_v2.o",
            )

            shims = [tile(col, 0) for col in range(COLS)]
            mems = [tile(col, 1) for col in range(COLS)]
            cores = [
                [tile(col, row + 2) for col in range(COLS)]
                for row in range(rows)
            ]

            mem_a = [None] * COLS
            in_a = [None] * COLS
            mem_w = [None] * COLS
            in_w = [[None] * COLS for _ in range(rows)]
            mem_o = [None] * COLS
            out_o = [[None] * COLS for _ in range(rows)]
            for col in range(COLS):
                destinations = [cores[row][col] for row in range(rows)]
                mem_a[col] = object_fifo(
                    f"q8p3mem_A_{col}", shims[col], mems[col], 2,
                    activation_ty,
                )
                in_a[col] = object_fifo(
                    f"q8p3in_A_{col}", mems[col], destinations, 2,
                    activation_ty,
                )
                object_fifo_link(mem_a[col], in_a[col])
                mem_w[col] = object_fifo(
                    f"q8p3mem_W_{col}", shims[col], mems[col], 1,
                    weight_mem_ty,
                )
                for row in range(rows):
                    in_w[row][col] = object_fifo(
                        f"q8p3in_W_{row}_{col}", mems[col], cores[row][col],
                        1, weight_ty,
                    )
                object_fifo_link(
                    mem_w[col], [in_w[row][col] for row in range(rows)], [],
                    [row * CORRECTED_TILE_BF16 for row in range(rows)],
                )
                mem_o[col] = object_fifo(
                    f"q8p3mem_O_{col}", mems[col], shims[col], 2,
                    output_mem_ty,
                )
                for row in range(rows):
                    out_o[row][col] = object_fifo(
                        f"q8p3out_O_{row}_{col}", cores[row][col], mems[col],
                        2, output_ty,
                    )
                object_fifo_link(
                    [out_o[row][col] for row in range(rows)], mem_o[col],
                    [row * BATCH * TILE_N for row in range(rows)],
                )

            for row in range(rows):
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
                                    activation = in_a[col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    weight = in_w[row][col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    gemm(activation, weight, result)
                                    in_w[row][col].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    in_a[col].release(
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
                        offsets=[0, 0, 0, col * BATCH * TILE_N],
                        sizes=[groups, 1, rows, BATCH * TILE_N],
                        strides=[BATCH * outputs_per_group, 0,
                                 COLS * BATCH * TILE_N, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=mem_a[col], bd_id=2, mem=host_input,
                        sizes=[groups, k_tiles, BATCH, TILE_K],
                        strides=[BATCH * k, TILE_K, k, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=mem_w[col], bd_id=3, mem=weights,
                        offsets=[
                            0, 0, 0,
                            col * groups * tasks_per_group *
                            task_stride_bf16,
                        ],
                        sizes=[groups * tasks_per_group, task_k_tiles, 1,
                               rows * CORRECTED_TILE_BF16],
                        strides=[
                            task_stride_bf16,
                            rows * CORRECTED_TILE_BF16, 0, 1,
                        ],
                    )
                dma_wait(*mem_o)

        print(ctx.module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)
    args = parser.parse_args()
    build_projection(args.K, args.N)
