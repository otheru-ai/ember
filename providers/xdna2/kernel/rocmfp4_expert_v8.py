# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Route-aware five-row ROCMFP4_FAST expert kernel for DSpark on XDNA2."""

import numpy as np
from ml_dtypes import bfloat16

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
V8_TILE_BYTES = TILE_K * TILE_N // 2 + (TILE_K // 32) * TILE_N * 2
PACKED_BYTES = (EMBD * FF + EMBD * FF + FF * EMBD) * 9 // 16
WEIGHT_COLUMN_BYTES = PACKED_BYTES // COLS
PACKET_BF16 = 4 * TILE_N


def build_expert():
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            activation_ty = np.ndarray[(BATCH * TILE_K,), np.dtype[bfloat16]]
            weight_ty = np.ndarray[(V8_TILE_BYTES,), np.dtype[np.uint8]]
            weight_mem_ty = np.ndarray[(ROWS * V8_TILE_BYTES,), np.dtype[np.uint8]]
            accum_ty = np.ndarray[(BATCH * TILE_N,), np.dtype[np.float32]]
            packet_ty = np.ndarray[(BATCH * PACKET_BF16,), np.dtype[bfloat16]]
            packet_mem_ty = np.ndarray[
                (ROWS * BATCH * PACKET_BF16,), np.dtype[bfloat16]
            ]

            zero = external_func("zero_rocmfp4_v8_f32x5", inputs=[accum_ty],
                                 link_with="rocmfp4_gemm_v8.o")
            gemm = external_func("gemm_rocmfp4_v8_f32x5",
                                 inputs=[activation_ty, weight_ty, accum_ty],
                                 link_with="rocmfp4_gemm_v8.o")
            gemm_masked = external_func("gemm_rocmfp4_v8_masked_f32x5",
                                        inputs=[activation_ty, weight_ty,
                                                accum_ty],
                                        link_with="rocmfp4_gemm_v8.o")
            swiglu = external_func("swiglu_rocmfp4_v8_f32x5",
                                   inputs=[accum_ty, accum_ty, activation_ty,
                                           packet_ty],
                                   link_with="rocmfp4_expert_v8.o")
            store_down = external_func("store_down_rocmfp4_v8_f32x5",
                                       inputs=[accum_ty, accum_ty, packet_ty],
                                       link_with="rocmfp4_expert_v8.o")

            shims = [tile(col, 0) for col in range(COLS)]
            mems = [tile(col, 1) for col in range(COLS)]
            cores = [[tile(col, row + 2) for col in range(COLS)]
                     for row in range(ROWS)]
            gate_acc = [[buffer(cores[row][col], accum_ty,
                                name=f"gate8_{row}_{col}") for col in range(COLS)]
                        for row in range(ROWS)]
            up_acc = [[buffer(cores[row][col], accum_ty,
                              name=f"up8_{row}_{col}") for col in range(COLS)]
                      for row in range(ROWS)]
            down_acc = [[[buffer(cores[row][col], accum_ty,
                                 name=f"down8_{group}_{row}_{col}")
                          for col in range(COLS)] for row in range(ROWS)]
                        for group in range(2)]

            mem_a = [None] * COLS
            mem_w = [None] * COLS
            in_a = [None] * COLS
            in_hidden = [None] * COLS
            in_w = [[None] * COLS for _ in range(ROWS)]
            mem_packet = [None] * COLS
            out_packet = [[None] * COLS for _ in range(ROWS)]

            for col in range(COLS):
                destinations = [cores[row][col] for row in range(ROWS)]
                mem_a[col] = object_fifo(f"mem_A8_{col}", shims[col], mems[col],
                                         2, activation_ty)
                in_a[col] = object_fifo(f"in_A8_{col}", mems[col], destinations,
                                        2, activation_ty)
                object_fifo_link(mem_a[col], in_a[col])
                in_hidden[col] = in_a[col]

                mem_w[col] = object_fifo(f"mem_W8_{col}", shims[col], mems[col],
                                         2, weight_mem_ty)
                for row in range(ROWS):
                    in_w[row][col] = object_fifo(
                        f"in_W8_{row}_{col}", mems[col], cores[row][col], 2,
                        weight_ty)
                object_fifo_link(mem_w[col], [in_w[row][col] for row in range(ROWS)],
                                 [], [row * V8_TILE_BYTES for row in range(ROWS)])

                mem_packet[col] = object_fifo(
                    f"mem_O8_{col}", mems[col], shims[col], 2, packet_mem_ty)
                for row in range(ROWS):
                    out_packet[row][col] = object_fifo(
                        f"out_O8_{row}_{col}", cores[row][col], mems[col], 2,
                        packet_ty)
                object_fifo_link([out_packet[row][col] for row in range(ROWS)],
                                 mem_packet[col],
                                 [row * BATCH * PACKET_BF16 for row in range(ROWS)])

            for row in range(ROWS):
                for col in range(COLS):
                    @core(cores[row][col])
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            hidden = out_packet[row][col].acquire(
                                ObjectFifoPort.Produce, 1)
                            zero(gate_acc[row][col])
                            zero(up_acc[row][col])
                            for _ in range_(EMBD // TILE_K):
                                activation = in_a[col].acquire(ObjectFifoPort.Consume, 1)
                                gate_weight = in_w[row][col].acquire(ObjectFifoPort.Consume, 1)
                                gemm(activation, gate_weight,
                                     gate_acc[row][col])
                                in_w[row][col].release(ObjectFifoPort.Consume, 1)
                                up_weight = in_w[row][col].acquire(ObjectFifoPort.Consume, 1)
                                gemm(activation, up_weight,
                                     up_acc[row][col])
                                in_w[row][col].release(ObjectFifoPort.Consume, 1)
                                in_a[col].release(ObjectFifoPort.Consume, 1)
                            params = in_a[col].acquire(ObjectFifoPort.Consume, 1)
                            swiglu(gate_acc[row][col], up_acc[row][col],
                                   params, hidden)
                            in_a[col].release(ObjectFifoPort.Consume, 1)
                            out_packet[row][col].release(ObjectFifoPort.Produce, 1)

                            for group in range(2):
                                zero(down_acc[group][row][col])
                            for group in range(2):
                                for _ in range_(FF // TILE_K):
                                    activation = in_hidden[col].acquire(
                                        ObjectFifoPort.Consume, 1)
                                    weight = in_w[row][col].acquire(
                                        ObjectFifoPort.Consume, 1)
                                    gemm_masked(activation, weight,
                                                down_acc[group][row][col])
                                    in_hidden[col].release(ObjectFifoPort.Consume, 1)
                                    in_w[row][col].release(ObjectFifoPort.Consume, 1)
                            result = out_packet[row][col].acquire(
                                ObjectFifoPort.Produce, 1)
                            store_down(down_acc[0][row][col],
                                       down_acc[1][row][col], result)
                            out_packet[row][col].release(ObjectFifoPort.Produce, 1)

            host_input_ty = np.ndarray[
                (BATCH * (EMBD + TILE_K),), np.dtype[bfloat16]
            ]
            host_weight_ty = np.ndarray[(PACKED_BYTES,), np.dtype[np.uint8]]
            host_staging_ty = np.ndarray[
                (ROWS * COLS * BATCH * PACKET_BF16,), np.dtype[bfloat16]
            ]

            @runtime_sequence(host_input_ty, host_weight_ty, host_staging_ty)
            def sequence(host_input, weights, staging):
                for col in range(COLS):
                    npu_dma_memcpy_nd(metadata=mem_packet[col], bd_id=0, mem=staging,
                        offsets=[0, 0, 0, col * BATCH * PACKET_BF16],
                        sizes=[1, 1, ROWS, BATCH * PACKET_BF16],
                        strides=[0, 0, COLS * BATCH * PACKET_BF16, 1])
                    npu_dma_memcpy_nd(metadata=mem_a[col], bd_id=2, mem=host_input,
                        sizes=[1, EMBD // TILE_K + 1, BATCH, TILE_K],
                        strides=[0, TILE_K, EMBD + TILE_K, 1])
                    npu_dma_memcpy_nd(metadata=mem_w[col], bd_id=3, mem=weights,
                        offsets=[0, 0, 0, col * WEIGHT_COLUMN_BYTES],
                        sizes=[1, 1, 1, PACKED_BYTES // COLS],
                        strides=[0, 0, 0, 1])
                dma_wait(*mem_packet)

                for _ in range(2):
                    for col in range(COLS):
                        npu_dma_memcpy_nd(metadata=mem_a[col], bd_id=2, mem=staging,
                            sizes=[FF // TILE_K, BATCH, 2, TILE_N],
                            strides=[2 * BATCH * PACKET_BF16,
                                     PACKET_BF16,
                                     BATCH * PACKET_BF16,
                                     1],
                            issue_token=True)
                    dma_wait(*mem_a)

                for col in range(COLS):
                    npu_dma_memcpy_nd(metadata=mem_packet[col], bd_id=0, mem=staging,
                        offsets=[0, 0, 0, col * BATCH * PACKET_BF16],
                        sizes=[1, 1, ROWS, BATCH * PACKET_BF16],
                        strides=[0, 0, COLS * BATCH * PACKET_BF16, 1])
                dma_wait(*mem_packet)

        print(ctx.module)


if __name__ == "__main__":
    build_expert()
