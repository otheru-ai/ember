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
OBJECT := $(BUILD_DIR)/rocmfp2_gemv.o
XCLBIN := $(BUILD_DIR)/gemv_$(K)x$(N).xclbin
INSTS := $(BUILD_DIR)/gemv_$(K)x$(N).insts
XCLBIN_V2 := $(BUILD_DIR)/gemv_v2_$(K)x$(N).xclbin
INSTS_V2 := $(BUILD_DIR)/gemv_v2_$(K)x$(N).insts

.PHONY: all clean
all: $(XCLBIN) $(XCLBIN_V2)

$(MLIR): rocmfp2_gemv.py
	mkdir -p $(dir $@)
	python3 $< -K $(K) -N $(N) > $@

$(MLIR_V2): rocmfp2_gemv.py
	mkdir -p $(dir $@)
	python3 $< -K $(K) -N $(N) --generation 2 > $@

$(OBJECT): rocmfp2_gemv.cc
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

clean:
	rm -rf build
