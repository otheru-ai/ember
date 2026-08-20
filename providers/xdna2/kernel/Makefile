# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Ember ships one XDNA2 overlay: the resident DSpark Q8 projection/shared-
# expert kernel. Historical target-MoE and reconfiguration experiments are
# documented in docs/xdna2-moe-prototype.md but are no longer built or shipped.
K ?= 4096
N ?= 2048
RESIDENT_MODE ?= projection
BUILD_DIR ?= build/$(K)x$(N)
MLIR_AIE_INSTALL_DIR ?= $(shell python3 -m pip show mlir-aie 2>/dev/null | awk '/^Location:/{print $$2 "/mlir_aie"}')
PEANO_INSTALL_DIR ?= $(shell python3 -m pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $$2 "/llvm-aie"}')
KERNEL_CC ?= $(PEANO_INSTALL_DIR)/bin/clang++
KERNEL_FLAGS ?= --target=aie2p-none-unknown-elf -Wno-unknown-attributes \
	-I$(MLIR_AIE_INSTALL_DIR)/include -D__AIE_API_AIE_ADF_HPP__ \
	-std=c++20 -O3 -DNDEBUG

OBJECT_RESIDENT := $(BUILD_DIR)/dspark_resident_v1.o
OBJECT_Q8 := $(BUILD_DIR)/q8_gemm_v2.o
OBJECT_EPILOGUE := $(BUILD_DIR)/rocmfp4_expert_v8.o
OBJECTS := $(OBJECT_RESIDENT) $(OBJECT_Q8) $(OBJECT_EPILOGUE)
MLIR := $(BUILD_DIR)/dspark_resident_v1.mlir
XCLBIN := $(BUILD_DIR)/dspark_resident_v1_$(RESIDENT_MODE)_$(K)x$(N)_b5.xclbin
INSTS := $(BUILD_DIR)/dspark_resident_v1_$(RESIDENT_MODE)_$(K)x$(N)_b5.insts

.PHONY: all dspark-resident-v1 clean
all: dspark-resident-v1
dspark-resident-v1: $(XCLBIN)

$(OBJECT_RESIDENT): dspark_resident_v1.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(OBJECT_Q8): q8_gemm_v2.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(OBJECT_EPILOGUE): rocmfp4_expert_v8.cc
	mkdir -p $(dir $@)
	$(KERNEL_CC) $(KERNEL_FLAGS) -c $< -o $@

$(MLIR): dspark_resident_v1.py
	mkdir -p $(dir $@)
	python3 $< --mode $(RESIDENT_MODE) -K $(K) -N $(N) > $@

$(XCLBIN): $(MLIR) $(OBJECTS)
	cd $(BUILD_DIR) && aiecc.py --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --no-compile-host \
		--xclbin-name=$(notdir $(XCLBIN)) --no-xchesscc --no-xbridge \
		--peano=$(PEANO_INSTALL_DIR) --aie-generate-npu-insts \
		--npu-insts-name=$(notdir $(INSTS)) $(notdir $(MLIR))
	test -s $(XCLBIN)
	test -s $(INSTS)

clean:
	rm -rf $(BUILD_DIR)
