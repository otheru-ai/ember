# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Resident four-row Q8 projection with streamed runtime loop bounds.

Every generated shape has the same AIE tile, FIFO, and core topology.  A
header object preceding the activation stream carries K-tile and output-group
counts, so a blocked core cannot begin the next invocation with stale RTP
state.  Shape-specific runtime sequences may therefore change shim DMA
descriptors without replacing the L1/L2 overlay.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects import arith
from aie.extras import types as T
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
ROWS = 4
OUTPUTS_PER_GROUP = ROWS * COLS * TILE_N
CORRECTED_TILE_BF16 = 2 * TILE_K * TILE_N
HEADER_BF16 = BATCH * TILE_K


def build_projection(k, n):
    if k <= 0 or k % 512:
        raise ValueError("K must be a positive multiple of 512")
    if n <= 0 or n % (COLS * TILE_N):
        raise ValueError("N must be a positive multiple of 512")
    padded_n = max(OUTPUTS_PER_GROUP, n)
    if padded_n % OUTPUTS_PER_GROUP:
        raise ValueError("N above 2048 must be a multiple of 2048")
    k_tiles = k // TILE_K
    groups = padded_n // OUTPUTS_PER_GROUP
    task_k_tiles = 4
    tasks_per_group = k_tiles // task_k_tiles
    task_payload_bf16 = task_k_tiles * ROWS * CORRECTED_TILE_BF16
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
                (ROWS * CORRECTED_TILE_BF16,), np.dtype[bfloat16]
            ]
            output_ty = np.ndarray[(BATCH * TILE_N,), np.dtype[np.float32]]
            output_mem_ty = np.ndarray[
                (ROWS * BATCH * TILE_N,), np.dtype[np.float32]
            ]

            zero = external_func(
                "zero_q8_v2_f32x5", inputs=[output_ty],
                link_with="q8_gemm_v2.o",
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
                for row in range(ROWS)
            ]

            mem_a = [None] * COLS
            in_a = [None] * COLS
            mem_w = [None] * COLS
            in_w = [[None] * COLS for _ in range(ROWS)]
            mem_o = [None] * COLS
            out_o = [[None] * COLS for _ in range(ROWS)]
            for col in range(COLS):
                destinations = [cores[row][col] for row in range(ROWS)]
                mem_a[col] = object_fifo(
                    f"q8u4mem_A_{col}", shims[col], mems[col], 2,
                    activation_ty,
                )
                in_a[col] = object_fifo(
                    f"q8u4in_A_{col}", mems[col], destinations, 2,
                    activation_ty,
                )
                object_fifo_link(mem_a[col], in_a[col])
                mem_w[col] = object_fifo(
                    f"q8u4mem_W_{col}", shims[col], mems[col], 1,
                    weight_mem_ty,
                )
                for row in range(ROWS):
                    in_w[row][col] = object_fifo(
                        f"q8u4in_W_{row}_{col}", mems[col], cores[row][col],
                        1, weight_ty,
                    )
                object_fifo_link(
                    mem_w[col], [in_w[row][col] for row in range(ROWS)], [],
                    [row * CORRECTED_TILE_BF16 for row in range(ROWS)],
                )
                mem_o[col] = object_fifo(
                    f"q8u4mem_O_{col}", mems[col], shims[col], 2,
                    output_mem_ty,
                )
                for row in range(ROWS):
                    out_o[row][col] = object_fifo(
                        f"q8u4out_O_{row}_{col}", cores[row][col], mems[col],
                        2, output_ty,
                    )
                object_fifo_link(
                    [out_o[row][col] for row in range(ROWS)], mem_o[col],
                    [row * BATCH * TILE_N for row in range(ROWS)],
                )

            for row in range(ROWS):
                for col in range(COLS):
                    @core(cores[row][col])
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            header = in_a[col].acquire(
                                ObjectFifoPort.Consume, 1
                            )
                            runtime_k_tiles = arith.fptosi(T.i32(), header[0])
                            runtime_groups = arith.fptosi(T.i32(), header[1])
                            in_a[col].release(ObjectFifoPort.Consume, 1)
                            for _ in range_(runtime_groups):
                                result = out_o[row][col].acquire(
                                    ObjectFifoPort.Produce, 1
                                )
                                zero(result)
                                for _ in range_(runtime_k_tiles):
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
                (HEADER_BF16 + groups * BATCH * k,), np.dtype[bfloat16]
            ]
            host_weight_ty = np.ndarray[(packed_bf16,), np.dtype[bfloat16]]
            host_output_ty = np.ndarray[
                (BATCH * padded_n,), np.dtype[np.float32]
            ]

            @runtime_sequence(host_input_ty, host_weight_ty, host_output_ty)
            def sequence(host_input, weights, output):
                for col in range(COLS):
                    npu_dma_memcpy_nd(
                        metadata=mem_o[col], bd_id=0, mem=output,
                        offsets=[0, 0, 0, col * BATCH * TILE_N],
                        sizes=[groups, 1, ROWS, BATCH * TILE_N],
                        strides=[BATCH * OUTPUTS_PER_GROUP, 0,
                                 COLS * BATCH * TILE_N, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=mem_a[col], bd_id=2, mem=host_input,
                        sizes=[1, 1, BATCH, TILE_K],
                        strides=[0, 0, TILE_K, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=mem_a[col], bd_id=4, mem=host_input,
                        offsets=[0, 0, 0, HEADER_BF16],
                        sizes=[groups, k_tiles, BATCH, TILE_K],
                        strides=[BATCH * k, TILE_K, k, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=mem_w[col], bd_id=3, mem=weights,
                        offsets=[
                            0, 0, 0,
                            col * groups * tasks_per_group * task_stride_bf16,
                        ],
                        sizes=[groups * tasks_per_group, task_k_tiles, 1,
                               ROWS * CORRECTED_TILE_BF16],
                        strides=[task_stride_bf16,
                                 ROWS * CORRECTED_TILE_BF16, 0, 1],
                    )
                dma_wait(*mem_o)

        print(ctx.module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)
    args = parser.parse_args()
    build_projection(args.K, args.N)
