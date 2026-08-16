# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

K ?= 4096
N ?= 2048
BUILD_DIR ?= build/$(K)x$(N)
MLIR_AIE_INSTALL_DIR ?= $(shell python3 -m pip show mlir-aie 2>/dev/null | awk '/^Location:/{print $$2 "/mlir_aie"}')
PEANO_INSTALL_DIR ?= $(shell python3 -m pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $$2 "/llvm-aie"}')
KERNEL_CC ?= $(PEANO_INSTALL_DIR)/bin/clang++
KERNEL_FLAGS ?= --target=aie2p-none-unknown-elf -Wno-unknown-attributes \
	-I$(MLIR_AIE_INSTALL_DIR)/include -D__AIE_API_AIE_ADF_HPP__ \
	-std=c++20 -O3 -DNDEBUG

MLIR := $(BUILD_DIR)/design.mlir
MLIR_V2 := $(BUILD_DIR)/design_v2.mlir
MLIR_V3 := $(BUILD_DIR)/design_v3.mlir
MLIR_V4 := $(BUILD_DIR)/design_v4.mlir
OBJECT := $(BUILD_DIR)/rocmfp2_gemv.o
OBJECT_V4 := $(BUILD_DIR)/rocmfp2_gemv_v4.o
OBJECT_EXPERT_V5 := $(BUILD_DIR)/rocmfp2_expert_v5.o
OBJECT_GEMM_V6 := $(BUILD_DIR)/rocmfp2_gemm_v6.o
OBJECT_EXPERT_V6 := $(BUILD_DIR)/rocmfp2_expert_v6.o
OBJECT_GEMM_V7 := $(BUILD_DIR)/rocmfp4_gemm_v7.o
OBJECT_EXPERT_V7 := $(BUILD_DIR)/rocmfp4_expert_v7.o
XCLBIN := $(BUILD_DIR)/gemv_$(K)x$(N).xclbin
INSTS := $(BUILD_DIR)/gemv_$(K)x$(N).insts
XCLBIN_V2 := $(BUILD_DIR)/gemv_v2_$(K)x$(N).xclbin
INSTS_V2 := $(BUILD_DIR)/gemv_v2_$(K)x$(N).insts
XCLBIN_V3 := $(BUILD_DIR)/gemv_v3_$(K)x$(N).xclbin
INSTS_V3 := $(BUILD_DIR)/gemv_v3_$(K)x$(N).insts
XCLBIN_V4 := $(BUILD_DIR)/gemv_v4_$(K)x$(N).xclbin
INSTS_V4 := $(BUILD_DIR)/gemv_v4_$(K)x$(N).insts
MLIR_EXPERT_V5 := $(BUILD_DIR)/expert_v5.mlir
XCLBIN_EXPERT_V5 := $(BUILD_DIR)/expert_v5_4096x2048x4096.xclbin
INSTS_EXPERT_V5 := $(BUILD_DIR)/expert_v5_4096x2048x4096.insts
MLIR_EXPERT_V6 := $(BUILD_DIR)/expert_v6.mlir
XCLBIN_EXPERT_V6 := $(BUILD_DIR)/expert_v6_4096x2048x4096_b4.xclbin
INSTS_EXPERT_V6 := $(BUILD_DIR)/expert_v6_4096x2048x4096_b4.insts
MLIR_EXPERT_V7 := $(BUILD_DIR)/expert_v7.mlir
XCLBIN_EXPERT_V7 := $(BUILD_DIR)/expert_v7_4096x2048x4096_b5.xclbin
INSTS_EXPERT_V7 := $(BUILD_DIR)/expert_v7_4096x2048x4096_b5.insts
MLIR_GATE_V5 := $(BUILD_DIR)/gate_v5.mlir
XCLBIN_GATE_V5 := $(BUILD_DIR)/gate_v5_4096x2048.xclbin
INSTS_GATE_V5 := $(BUILD_DIR)/gate_v5_4096x2048.insts
MLIR_ONE_DOWN_V5 := $(BUILD_DIR)/one_down_v5.mlir
XCLBIN_ONE_DOWN_V5 := $(BUILD_DIR)/one_down_v5_4096x2048.xclbin
INSTS_ONE_DOWN_V5 := $(BUILD_DIR)/one_down_v5_4096x2048.insts

.PHONY: all expert-v5 expert-v6 expert-v7 gate-v5 one-down-v5 clean
all: $(XCLBIN) $(XCLBIN_V2) $(XCLBIN_V3) $(XCLBIN_V4)
expert-v5: $(XCLBIN_EXPERT_V5)
expert-v6: $(XCLBIN_EXPERT_V6)
expert-v7: $(XCLBIN_EXPERT_V7)
gate-v5: $(XCLBIN_GATE_V5)
one-down-v5: $(XCLBIN_ONE_DOWN_V5)

$(MLIR): rocmfp2_gemv.py
	mkdir -p $(dir $@)
	python3 $< -K $(K) -N $(N) > $@

$(MLIR_V2): rocmfp2_gemv.py
	mkdir -p $(dir $@)
	python3 $< -K $(K) -N $(N) --generation 2 > $@

$(MLIR_V3): rocmfp2_gemv.py
	mkdir -p $(dir $@)
	python3 $< -K $(K) -N $(N) --generation 3 > $@

$(MLIR_V4): rocmfp2_gemv.py
	mkdir -p $(dir $@)
	python3 $< -K $(K) -N $(N) --generation 4 > $@

$(OBJECT): rocmfp2_gemv.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(OBJECT_V4): rocmfp2_gemv_v4.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(OBJECT_EXPERT_V5): rocmfp2_expert_v5.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(OBJECT_GEMM_V6): rocmfp2_gemm_v6.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(OBJECT_EXPERT_V6): rocmfp2_expert_v6.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(OBJECT_GEMM_V7): rocmfp4_gemm_v7.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(OBJECT_EXPERT_V7): rocmfp4_expert_v7.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(MLIR_EXPERT_V5): rocmfp2_expert_v5.py
	mkdir -p $(dir $@)
	python3 $< > $@

$(MLIR_EXPERT_V6): rocmfp2_expert_v6.py
	mkdir -p $(dir $@)
	python3 $< > $@

$(MLIR_EXPERT_V7): rocmfp4_expert_v7.py
	mkdir -p $(dir $@)
	python3 $< > $@

$(MLIR_GATE_V5): rocmfp2_expert_v5.py
	mkdir -p $(dir $@)
	python3 $< --gate-only > $@

$(MLIR_ONE_DOWN_V5): rocmfp2_expert_v5.py
	mkdir -p $(dir $@)
	python3 $< --one-down > $@

$(XCLBIN): $(MLIR) $(OBJECT)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS)) $(notdir $(MLIR))
	test -s $(XCLBIN)
	test -s $(INSTS)

$(XCLBIN_V2): $(MLIR_V2) $(OBJECT)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN_V2)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS_V2)) $(notdir $(MLIR_V2))
	test -s $(XCLBIN_V2)
	test -s $(INSTS_V2)

$(XCLBIN_V3): $(MLIR_V3) $(OBJECT)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN_V3)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS_V3)) $(notdir $(MLIR_V3))
	test -s $(XCLBIN_V3)
	test -s $(INSTS_V3)

$(XCLBIN_V4): $(MLIR_V4) $(OBJECT_V4)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN_V4)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS_V4)) $(notdir $(MLIR_V4))
	test -s $(XCLBIN_V4)
	test -s $(INSTS_V4)

$(XCLBIN_EXPERT_V5): $(MLIR_EXPERT_V5) $(OBJECT_V4) $(OBJECT_EXPERT_V5)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN_EXPERT_V5)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS_EXPERT_V5)) $(notdir $(MLIR_EXPERT_V5))
	test -s $(XCLBIN_EXPERT_V5)
	test -s $(INSTS_EXPERT_V5)

$(XCLBIN_EXPERT_V6): $(MLIR_EXPERT_V6) $(OBJECT_GEMM_V6) $(OBJECT_EXPERT_V6)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN_EXPERT_V6)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS_EXPERT_V6)) $(notdir $(MLIR_EXPERT_V6))
	test -s $(XCLBIN_EXPERT_V6)
	test -s $(INSTS_EXPERT_V6)

$(XCLBIN_EXPERT_V7): $(MLIR_EXPERT_V7) $(OBJECT_GEMM_V7) $(OBJECT_EXPERT_V7)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN_EXPERT_V7)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS_EXPERT_V7)) $(notdir $(MLIR_EXPERT_V7))
	test -s $(XCLBIN_EXPERT_V7)
	test -s $(INSTS_EXPERT_V7)

$(XCLBIN_GATE_V5): $(MLIR_GATE_V5) $(OBJECT_V4) $(OBJECT_EXPERT_V5)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN_GATE_V5)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS_GATE_V5)) $(notdir $(MLIR_GATE_V5))
	test -s $(XCLBIN_GATE_V5)
	test -s $(INSTS_GATE_V5)

$(XCLBIN_ONE_DOWN_V5): $(MLIR_ONE_DOWN_V5) $(OBJECT_V4) $(OBJECT_EXPERT_V5)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN_ONE_DOWN_V5)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS_ONE_DOWN_V5)) $(notdir $(MLIR_ONE_DOWN_V5))
	test -s $(XCLBIN_ONE_DOWN_V5)
	test -s $(INSTS_ONE_DOWN_V5)

clean:
	rm -rf build
