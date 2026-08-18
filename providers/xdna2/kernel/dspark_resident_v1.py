# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Resident DSpark Q8 projection/shared-expert overlay for XDNA2.

The AIE core, FIFO, and memory topology is invariant. BF16-exact streamed
header loop counts activate one of two paths for each dispatch:

  0: dynamic Q8 projection
  1: fixed 4096x2048x4096 Q8 shared expert
Mode-specific runtime sequences change only shim DMA descriptors and host BO
sizes. This follows the AIE-RT descriptor-queue model: the array stays resident
and cores block on the next header instead of being reconfigured by the host.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects import arith
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import (
    AIEDevice, ObjectFifoPort, buffer, core, device, external_func,
    object_fifo, object_fifo_link, tile,
)
from aie.dialects.aiex import dma_wait, npu_dma_memcpy_nd, runtime_sequence
from aie.helpers.dialects.scf import _for as range_


BATCH = 5
TILE_K = 128
TILE_N = 64
ROWS = 4
COLS = 8
EMBD = 4096
FF = 2048
OUTPUTS_PER_GROUP = ROWS * COLS * TILE_N
HEADER_BF16 = BATCH * TILE_K
Q8_TILE_BYTES = 4 * TILE_K * TILE_N
Q8_EXPERT_BYTES = (EMBD * FF + EMBD * FF + FF * EMBD) * 4
PACKET_BF16 = 4 * TILE_N


def build_resident(mode, k, n):
    if mode not in ("projection", "shared"):
        raise ValueError("mode must be projection or shared")
    if k <= 0 or k % 512:
        raise ValueError("K must be a positive multiple of 512")
    if n <= 0 or n % 512:
        raise ValueError("N must be a positive multiple of 512")
    padded_n = max(OUTPUTS_PER_GROUP, n)
    if padded_n % OUTPUTS_PER_GROUP:
        raise ValueError("projection N above 2048 must be a multiple of 2048")
    projection_k_tiles = k // TILE_K
    projection_groups = padded_n // OUTPUTS_PER_GROUP
    task_k_tiles = 4
    tasks_per_group = projection_k_tiles // task_k_tiles
    q8_task_bytes = task_k_tiles * ROWS * Q8_TILE_BYTES + 4
    projection_q8_bytes = (
        COLS * projection_groups * tasks_per_group * q8_task_bytes
    )

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            activation_ty = np.ndarray[
                (BATCH * TILE_K,), np.dtype[bfloat16]
            ]
            weight_ty = np.ndarray[(Q8_TILE_BYTES,), np.dtype[np.uint8]]
            weight_mem_ty = np.ndarray[
                (ROWS * Q8_TILE_BYTES,), np.dtype[np.uint8]
            ]
            accum_ty = np.ndarray[(BATCH * TILE_N,), np.dtype[np.float32]]
            projection_mem_ty = np.ndarray[
                (ROWS * BATCH * TILE_N,), np.dtype[np.float32]
            ]
            packet_ty = np.ndarray[
                (BATCH * PACKET_BF16,), np.dtype[bfloat16]
            ]
            packet_mem_ty = np.ndarray[
                (ROWS * BATCH * PACKET_BF16,), np.dtype[bfloat16]
            ]

            zero_q8 = external_func(
                "zero_q8_v2_f32x5", inputs=[accum_ty],
                link_with="q8_gemm_v2.o",
            )
            gemm_q8 = external_func(
                "gemm_q8_v2_f32x5",
                inputs=[activation_ty, weight_ty, accum_ty],
                link_with="q8_gemm_v2.o",
            )
            swiglu = external_func(
                "swiglu_rocmfp4_v8_f32x5",
                inputs=[accum_ty, accum_ty, activation_ty, packet_ty],
                link_with="rocmfp4_expert_v8.o",
            )
            store_down = external_func(
                "store_down_rocmfp4_v8_f32x5",
                inputs=[accum_ty, accum_ty, packet_ty],
                link_with="rocmfp4_expert_v8.o",
            )
            store_projection = external_func(
                "store_projection_resident_f32x5",
                inputs=[accum_ty, packet_ty],
                link_with="dspark_resident_v1.o",
            )

            shims = [tile(col, 0) for col in range(COLS)]
            mems = [tile(col, 1) for col in range(COLS)]
            cores = [
                [tile(col, row + 2) for col in range(COLS)]
                for row in range(ROWS)
            ]
            gate_acc = [[
                buffer(cores[row][col], accum_ty,
                       name=f"resident_gate_{row}_{col}")
                for col in range(COLS)
            ] for row in range(ROWS)]
            up_acc = [[
                buffer(cores[row][col], accum_ty,
                       name=f"resident_up_{row}_{col}")
                for col in range(COLS)
            ] for row in range(ROWS)]
            down_acc = [[[
                buffer(cores[row][col], accum_ty,
                       name=f"resident_down_{group}_{row}_{col}")
                for col in range(COLS)
            ] for row in range(ROWS)] for group in range(2)]
            projection_acc = [[
                buffer(cores[row][col], accum_ty,
                       name=f"resident_projection_{row}_{col}")
                for col in range(COLS)
            ] for row in range(ROWS)]

            mem_a = [None] * COLS
            in_a = [None] * COLS
            mem_weight = [None] * COLS
            in_weight = [[None] * COLS for _ in range(ROWS)]
            mem_packet = [None] * COLS
            out_packet = [[None] * COLS for _ in range(ROWS)]

            for col in range(COLS):
                destinations = [cores[row][col] for row in range(ROWS)]
                mem_a[col] = object_fifo(
                    f"resident_mem_A_{col}", shims[col], mems[col], 2,
                    activation_ty,
                )
                in_a[col] = object_fifo(
                    f"resident_in_A_{col}", mems[col], destinations, 2,
                    activation_ty,
                )
                object_fifo_link(mem_a[col], in_a[col])

                mem_weight[col] = object_fifo(
                    f"resident_mem_W_{col}", shims[col], mems[col], 1,
                    weight_mem_ty,
                )
                for row in range(ROWS):
                    in_weight[row][col] = object_fifo(
                        f"resident_in_W_{row}_{col}", mems[col],
                        cores[row][col], 1, weight_ty,
                    )
                object_fifo_link(
                    mem_weight[col],
                    [in_weight[row][col] for row in range(ROWS)], [],
                    [row * Q8_TILE_BYTES for row in range(ROWS)],
                )

                mem_packet[col] = object_fifo(
                    f"resident_mem_EO_{col}", mems[col], shims[col], 2,
                    packet_mem_ty,
                )
                for row in range(ROWS):
                    out_packet[row][col] = object_fifo(
                        f"resident_out_EO_{row}_{col}", cores[row][col],
                        mems[col], 2, packet_ty,
                    )
                object_fifo_link(
                    [out_packet[row][col] for row in range(ROWS)],
                    mem_packet[col],
                    [row * BATCH * PACKET_BF16 for row in range(ROWS)],
                )

            for row in range(ROWS):
                for col in range(COLS):
                    @core(cores[row][col])
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            header = in_a[col].acquire(
                                ObjectFifoPort.Consume, 1
                            )
                            runtime_k_tiles = arith.fptosi(T.i32(), header[1])
                            projection_runs = arith.fptosi(T.i32(), header[2])
                            shared_runs = arith.fptosi(T.i32(), header[3])
                            in_a[col].release(ObjectFifoPort.Consume, 1)

                            for _ in range_(projection_runs):
                                result = out_packet[row][col].acquire(
                                    ObjectFifoPort.Produce, 1
                                )
                                zero_q8(projection_acc[row][col])
                                for _ in range_(runtime_k_tiles):
                                    activation = in_a[col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    weight = in_weight[row][col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    gemm_q8(
                                        activation, weight,
                                        projection_acc[row][col]
                                    )
                                    in_weight[row][col].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    in_a[col].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                store_projection(
                                    projection_acc[row][col], result
                                )
                                out_packet[row][col].release(
                                    ObjectFifoPort.Produce, 1
                                )

                            for _ in range_(shared_runs):
                                hidden = out_packet[row][col].acquire(
                                    ObjectFifoPort.Produce, 1
                                )
                                zero_q8(gate_acc[row][col])
                                zero_q8(up_acc[row][col])
                                for _ in range_(EMBD // TILE_K):
                                    activation = in_a[col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    gate_weight = in_weight[row][col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    gemm_q8(activation, gate_weight,
                                            gate_acc[row][col])
                                    in_weight[row][col].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    up_weight = in_weight[row][col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    gemm_q8(activation, up_weight,
                                            up_acc[row][col])
                                    in_weight[row][col].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    in_a[col].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                params = in_a[col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                swiglu(gate_acc[row][col], up_acc[row][col],
                                       params, hidden)
                                in_a[col].release(ObjectFifoPort.Consume, 1)
                                out_packet[row][col].release(
                                    ObjectFifoPort.Produce, 1
                                )

                                for group in range(2):
                                    zero_q8(down_acc[group][row][col])
                                for group in range(2):
                                    for _ in range_(FF // TILE_K):
                                        activation = in_a[col].acquire(
                                            ObjectFifoPort.Consume, 1
                                        )
                                        weight = in_weight[row][col].acquire(
                                            ObjectFifoPort.Consume, 1
                                        )
                                        gemm_q8(
                                            activation, weight,
                                            down_acc[group][row][col],
                                        )
                                        in_weight[row][col].release(
                                            ObjectFifoPort.Consume, 1
                                        )
                                        in_a[col].release(
                                            ObjectFifoPort.Consume, 1
                                        )
                                result = out_packet[row][col].acquire(
                                    ObjectFifoPort.Produce, 1
                                )
                                store_down(
                                    down_acc[0][row][col],
                                    down_acc[1][row][col], result,
                                )
                                out_packet[row][col].release(
                                    ObjectFifoPort.Produce, 1
                                )

            if mode == "projection":
                input_elements = (
                    HEADER_BF16 + projection_groups * BATCH * k
                )
                q8_elements = projection_q8_bytes
                projection_elements = (
                    projection_groups * ROWS * COLS *
                    BATCH * PACKET_BF16
                )
                staging_elements = 1
            else:
                input_elements = HEADER_BF16 + BATCH * (EMBD + TILE_K)
                q8_elements = Q8_EXPERT_BYTES
                projection_elements = 1
                staging_elements = ROWS * COLS * BATCH * PACKET_BF16

            host_input_ty = np.ndarray[
                (input_elements,), np.dtype[bfloat16]
            ]
            host_q8_ty = np.ndarray[(q8_elements,), np.dtype[np.uint8]]
            host_projection_ty = np.ndarray[
                (projection_elements,), np.dtype[bfloat16]
            ]
            host_staging_ty = np.ndarray[
                (staging_elements,), np.dtype[bfloat16]
            ]

            @runtime_sequence(
                host_input_ty, host_q8_ty, host_projection_ty, host_staging_ty,
            )
            def sequence(
                    host_input, q8_weights, projection_output, staging):
                if mode == "projection":
                    for col in range(COLS):
                        npu_dma_memcpy_nd(
                            metadata=mem_packet[col], bd_id=0,
                            mem=projection_output,
                            offsets=[0, 0, 0,
                                     col * BATCH * PACKET_BF16],
                            sizes=[projection_groups, 1, ROWS,
                                   BATCH * PACKET_BF16],
                            strides=[ROWS * COLS * BATCH * PACKET_BF16, 0,
                                     COLS * BATCH * PACKET_BF16, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=mem_a[col], bd_id=2, mem=host_input,
                            sizes=[1, 1, BATCH, TILE_K],
                            strides=[0, 0, TILE_K, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=mem_a[col], bd_id=4, mem=host_input,
                            offsets=[0, 0, 0, HEADER_BF16],
                            sizes=[projection_groups, projection_k_tiles,
                                   BATCH, TILE_K],
                            strides=[BATCH * k, TILE_K, k, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=mem_weight[col], bd_id=3, mem=q8_weights,
                            offsets=[
                                0, 0, 0,
                                col * projection_groups * tasks_per_group *
                                q8_task_bytes,
                            ],
                            sizes=[projection_groups * tasks_per_group,
                                   task_k_tiles, 1,
                                   ROWS * Q8_TILE_BYTES],
                            strides=[q8_task_bytes,
                                     ROWS * Q8_TILE_BYTES, 0, 1],
                        )
                    dma_wait(*mem_packet)
                else:
                    for col in range(COLS):
                        npu_dma_memcpy_nd(
                            metadata=mem_packet[col], bd_id=0, mem=staging,
                            offsets=[0, 0, 0,
                                     col * BATCH * PACKET_BF16],
                            sizes=[1, 1, ROWS, BATCH * PACKET_BF16],
                            strides=[0, 0,
                                     COLS * BATCH * PACKET_BF16, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=mem_a[col], bd_id=2, mem=host_input,
                            sizes=[1, 1, BATCH, TILE_K],
                            strides=[0, 0, TILE_K, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=mem_a[col], bd_id=4, mem=host_input,
                            offsets=[0, 0, 0, HEADER_BF16],
                            sizes=[1, EMBD // TILE_K + 1, BATCH, TILE_K],
                            strides=[0, TILE_K, EMBD + TILE_K, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=mem_weight[col], bd_id=3,
                            mem=q8_weights,
                            offsets=[0, 0, 0,
                                     col * Q8_EXPERT_BYTES // COLS],
                            sizes=[1, 1, 1, Q8_EXPERT_BYTES // COLS],
                            strides=[0, 0, 0, 1],
                        )
                    dma_wait(*mem_packet)

                    for _ in range(2):
                        for col in range(COLS):
                            npu_dma_memcpy_nd(
                                metadata=mem_a[col], bd_id=2, mem=staging,
                                sizes=[FF // TILE_K, BATCH, 2, TILE_N],
                                strides=[2 * BATCH * PACKET_BF16,
                                         PACKET_BF16,
                                         BATCH * PACKET_BF16, 1],
                                issue_token=True,
                            )
                        dma_wait(*mem_a)

                    for col in range(COLS):
                        npu_dma_memcpy_nd(
                            metadata=mem_packet[col], bd_id=0, mem=staging,
                            offsets=[0, 0, 0,
                                     col * BATCH * PACKET_BF16],
                            sizes=[1, 1, ROWS, BATCH * PACKET_BF16],
                            strides=[0, 0,
                                     COLS * BATCH * PACKET_BF16, 1],
                        )
                    dma_wait(*mem_packet)

        print(ctx.module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("projection", "shared"),
                        required=True)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("-N", type=int, default=1024)
    args = parser.parse_args()
    build_resident(args.mode, args.K, args.N)
