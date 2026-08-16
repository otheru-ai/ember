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
XCLBIN := $(BUILD_DIR)/gemv_$(K)x$(N).xclbin
INSTS := $(BUILD_DIR)/gemv_$(K)x$(N).insts
XCLBIN_V2 := $(BUILD_DIR)/gemv_v2_$(K)x$(N).xclbin
INSTS_V2 := $(BUILD_DIR)/gemv_v2_$(K)x$(N).insts
XCLBIN_V3 := $(BUILD_DIR)/gemv_v3_$(K)x$(N).xclbin
INSTS_V3 := $(BUILD_DIR)/gemv_v3_$(K)x$(N).insts
XCLBIN_V4 := $(BUILD_DIR)/gemv_v4_$(K)x$(N).xclbin
INSTS_V4 := $(BUILD_DIR)/gemv_v4_$(K)x$(N).insts

.PHONY: all clean
all: $(XCLBIN) $(XCLBIN_V2) $(XCLBIN_V3) $(XCLBIN_V4)

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

clean:
	rm -rf build
