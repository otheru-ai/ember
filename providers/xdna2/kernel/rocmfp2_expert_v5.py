# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Fixed 4096x2048x4096 spatially fused ROCMFP2 expert for XDNA2.

One resident ctrlcode performs gate and up GEMVs, exact clamped SwiGLU, and
both down-projection array passes.  The hidden BF16 BO is an NPU-owned staging
surface inside the same command; the CPU never observes or fences it.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import (
    AIEDevice, ObjectFifoPort, buffer, core, device, external_func,
    object_fifo, object_fifo_link, tile,
)
from aie.dialects.aiex import dma_wait, npu_dma_memcpy_nd, runtime_sequence
from aie.helpers.dialects.scf import _for as range_


TILE_K = 128
TILE_N = 64
ROWS = 4
COLS = 8
EMBD = 4096
FF = 2048
V4_TILE_BYTES = TILE_K * TILE_N // 2 + 2 * (TILE_K // 32) * TILE_N * 2
PACKED_BYTES = (EMBD * FF + EMBD * FF + FF * EMBD) * 10 // 16
WEIGHT_COLUMN_BYTES = PACKED_BYTES // COLS


def build_expert(gate_only=False, down_groups=2):
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            activation_ty = np.ndarray[(TILE_K,), np.dtype[bfloat16]]
            weight_ty = np.ndarray[(V4_TILE_BYTES,), np.dtype[np.uint8]]
            weight_mem_ty = np.ndarray[(ROWS * V4_TILE_BYTES,), np.dtype[np.uint8]]
            accum_ty = np.ndarray[(TILE_N,), np.dtype[np.float32]]
            # One 512-byte packet can carry either 64 BF16 hidden values (plus
            # padding) or the core's two 64-lane FP32 down results.
            packet_ty = np.ndarray[(4 * TILE_N,), np.dtype[bfloat16]]
            packet_mem_ty = np.ndarray[(ROWS * 4 * TILE_N,), np.dtype[bfloat16]]

            zero = external_func("zero_rocmfp2_v4_f32", inputs=[accum_ty],
                                 link_with="rocmfp2_gemv_v4.o")
            gemv = external_func("gemv_rocmfp2_v4_f32",
                                 inputs=[activation_ty, weight_ty, accum_ty],
                                 link_with="rocmfp2_gemv_v4.o")
            swiglu = external_func("swiglu_rocmfp2_v5",
                                   inputs=[accum_ty, accum_ty, activation_ty, packet_ty],
                                   link_with="rocmfp2_expert_v5.o")
            store_down = external_func("store_down_rocmfp2_v5",
                                       inputs=[accum_ty, accum_ty, packet_ty],
                                       link_with="rocmfp2_expert_v5.o")

            shims = [tile(col, 0) for col in range(COLS)]
            mems = [tile(col, 1) for col in range(COLS)]
            cores = [[tile(col, row + 2) for col in range(COLS)]
                     for row in range(ROWS)]
            gate_acc = [[buffer(cores[row][col], accum_ty,
                                name=f"gate_{row}_{col}") for col in range(COLS)]
                        for row in range(ROWS)]
            up_acc = [[buffer(cores[row][col], accum_ty,
                              name=f"up_{row}_{col}") for col in range(COLS)]
                      for row in range(ROWS)]
            down_acc = [[[buffer(cores[row][col], accum_ty,
                                 name=f"down_{group}_{row}_{col}")
                          for col in range(COLS)] for row in range(ROWS)]
                        for group in range(2)]

            mem_a, mem_w = [None] * COLS, [None] * COLS
            in_a = [None] * COLS
            in_w = [[None] * COLS for _ in range(ROWS)]
            in_hidden = [None] * COLS
            mem_packet = [None] * COLS
            out_packet = [[None] * COLS for _ in range(ROWS)]

            for col in range(COLS):
                destinations = [cores[row][col] for row in range(ROWS)]
                mem_a[col] = object_fifo(f"mem_A{col}", shims[col], mems[col],
                                         2, activation_ty)
                in_a[col] = object_fifo(f"in_A{col}", mems[col], destinations,
                                        2, activation_ty)
                object_fifo_link(mem_a[col], in_a[col])
                mem_w[col] = object_fifo(f"mem_W{col}", shims[col], mems[col],
                                         2, weight_mem_ty)
                for row in range(ROWS):
                    in_w[row][col] = object_fifo(
                        f"in_W{row}_{col}", mems[col], cores[row][col], 2, weight_ty)
                object_fifo_link(mem_w[col], [in_w[row][col] for row in range(ROWS)],
                                 [], [row * V4_TILE_BYTES for row in range(ROWS)])

                # The original activation, one parameter tile, and both hidden
                # replay passes share this BF16 stream.  That keeps each shim
                # within its two host-to-array DMA channels (activation/weights).
                in_hidden[col] = in_a[col]

                mem_packet[col] = object_fifo(
                    f"mem_O{col}", mems[col], shims[col], 2, packet_mem_ty)
                for row in range(ROWS):
                    out_packet[row][col] = object_fifo(
                        f"out_O{row}_{col}", cores[row][col], mems[col], 2, packet_ty)
                object_fifo_link([out_packet[row][col] for row in range(ROWS)],
                                 mem_packet[col],
                                 [row * 4 * TILE_N for row in range(ROWS)])

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
                                gemv(activation, gate_weight, gate_acc[row][col])
                                in_w[row][col].release(ObjectFifoPort.Consume, 1)
                                up_weight = in_w[row][col].acquire(ObjectFifoPort.Consume, 1)
                                gemv(activation, up_weight, up_acc[row][col])
                                in_w[row][col].release(ObjectFifoPort.Consume, 1)
                                in_a[col].release(ObjectFifoPort.Consume, 1)
                            params = in_a[col].acquire(ObjectFifoPort.Consume, 1)
                            swiglu(gate_acc[row][col], up_acc[row][col], params, hidden)
                            in_a[col].release(ObjectFifoPort.Consume, 1)
                            out_packet[row][col].release(ObjectFifoPort.Produce, 1)
                            if gate_only:
                                continue
                            for group in range(2):
                                zero(down_acc[group][row][col])
                            for group in range(down_groups):
                                for _ in range_(FF // TILE_K):
                                    activation = in_hidden[col].acquire(
                                        ObjectFifoPort.Consume, 1)
                                    weight = in_w[row][col].acquire(
                                        ObjectFifoPort.Consume, 1)
                                    gemv(activation, weight,
                                         down_acc[group][row][col])
                                    in_hidden[col].release(ObjectFifoPort.Consume, 1)
                                    in_w[row][col].release(ObjectFifoPort.Consume, 1)
                            result = out_packet[row][col].acquire(
                                ObjectFifoPort.Produce, 1)
                            store_down(down_acc[0][row][col],
                                       down_acc[1][row][col], result)
                            out_packet[row][col].release(ObjectFifoPort.Produce, 1)

            host_input_ty = np.ndarray[(EMBD + TILE_K,), np.dtype[bfloat16]]
            host_weight_ty = np.ndarray[(PACKED_BYTES,), np.dtype[np.uint8]]
            # 32 padded core packets. It first holds hidden BF16 payloads, then
            # is safely reused for the final two FP32 output groups per core.
            host_staging_ty = np.ndarray[(ROWS * COLS * 4 * TILE_N,),
                                         np.dtype[bfloat16]]

            @runtime_sequence(host_input_ty, host_weight_ty, host_staging_ty)
            def sequence(host_input, weights, staging):
                for col in range(COLS):
                    npu_dma_memcpy_nd(metadata=mem_packet[col], bd_id=0, mem=staging,
                        offsets=[0, 0, 0, col * 4 * TILE_N],
                        sizes=[1, 1, ROWS, 4 * TILE_N],
                        strides=[0, 0, COLS * 4 * TILE_N, 1])
                    npu_dma_memcpy_nd(metadata=mem_a[col], bd_id=2, mem=host_input,
                        sizes=[1, 1, 1, EMBD + TILE_K], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=mem_w[col], bd_id=3, mem=weights,
                        offsets=[0, 0, 0, col * WEIGHT_COLUMN_BYTES],
                        sizes=[1, 1, 1,
                               ((2 * EMBD * FF +
                                 (0 if gate_only else down_groups * FF * FF)) *
                                10 // 16 // COLS)],
                        strides=[0, 0, 0, 1])
                dma_wait(*mem_packet)
                if gate_only:
                    return
                for group in range(down_groups):
                    for col in range(COLS):
                        npu_dma_memcpy_nd(metadata=mem_a[col],
                            bd_id=2, mem=staging,
                            sizes=[1, 1, ROWS * COLS, TILE_N],
                            strides=[0, 0, 4 * TILE_N, 1],
                            # AIE-RT release/main_aig:
                            # XAie_DmaChannelSetStartQueue() makes completion
                            # token issuance a property of every queued task.
                            # dma_wait consumes that token, so descriptor reuse
                            # must request a new one rather than inheriting the
                            # token state of the original host-input transfer.
                            issue_token=True)
                    # AIE-RT task queues can chain both replays, but waiting
                    # each phase makes the producer/consumer dependency
                    # explicit and avoids filling the shallow broadcast FIFO.
                    dma_wait(*mem_a)
                for col in range(COLS):
                    npu_dma_memcpy_nd(metadata=mem_packet[col], bd_id=0, mem=staging,
                        offsets=[0, 0, 0, col * 4 * TILE_N],
                        sizes=[1, 1, ROWS, 4 * TILE_N],
                        strides=[0, 0, COLS * 4 * TILE_N, 1])
                dma_wait(*mem_packet)

        print(ctx.module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--one-down", action="store_true")
    args = parser.parse_args()
    build_expert(args.gate_only, 1 if args.one_down else 2)
