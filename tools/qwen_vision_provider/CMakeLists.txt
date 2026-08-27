cmake_minimum_required(VERSION 3.22)

project(ember_qwen4exp_vision_provider LANGUAGES CXX)

if(NOT DEFINED EMBER_SOURCE_DIR OR NOT IS_DIRECTORY "${EMBER_SOURCE_DIR}")
  message(FATAL_ERROR "EMBER_SOURCE_DIR must name the Ember source tree")
endif()
if(NOT DEFINED LLAMA_CPP_SOURCE_DIR OR NOT IS_DIRECTORY "${LLAMA_CPP_SOURCE_DIR}")
  message(FATAL_ERROR "LLAMA_CPP_SOURCE_DIR must name the pinned llama.cpp checkout")
endif()
if(NOT DEFINED LLAMA_CPP_INSTALL_DIR OR NOT IS_DIRECTORY "${LLAMA_CPP_INSTALL_DIR}")
  message(FATAL_ERROR "LLAMA_CPP_INSTALL_DIR must name the llama.cpp install prefix")
endif()

find_library(MTMD_LIBRARY NAMES mtmd
  PATHS "${LLAMA_CPP_INSTALL_DIR}/lib" "${LLAMA_CPP_INSTALL_DIR}/lib64"
  NO_DEFAULT_PATH REQUIRED)
find_library(LLAMA_LIBRARY NAMES llama
  PATHS "${LLAMA_CPP_INSTALL_DIR}/lib" "${LLAMA_CPP_INSTALL_DIR}/lib64"
  NO_DEFAULT_PATH REQUIRED)

add_library(ember_qwen4exp_vision_provider SHARED
  "${EMBER_SOURCE_DIR}/tools/qwen4exp_vision_provider_llamacpp.cpp")
target_compile_features(ember_qwen4exp_vision_provider PRIVATE cxx_std_17)
target_compile_options(ember_qwen4exp_vision_provider PRIVATE
  -Wall -Wextra -Werror -Wshadow -Wconversion -Wsign-conversion)
target_include_directories(ember_qwen4exp_vision_provider PRIVATE
  "${EMBER_SOURCE_DIR}/engine/dflash/qwen4exp")
target_include_directories(ember_qwen4exp_vision_provider SYSTEM PRIVATE
  "${LLAMA_CPP_SOURCE_DIR}/include"
  "${LLAMA_CPP_SOURCE_DIR}/ggml/include"
  "${LLAMA_CPP_SOURCE_DIR}/tools/mtmd")
target_link_libraries(ember_qwen4exp_vision_provider PRIVATE
  "${MTMD_LIBRARY}" "${LLAMA_LIBRARY}")
target_link_options(ember_qwen4exp_vision_provider PRIVATE
  "LINKER:--no-undefined"
  # The upstream shared libraries have no install RUNPATH. DT_RPATH is
  # intentionally inherited by their private ggml dependency closure.
  "LINKER:--disable-new-dtags")
set_target_properties(ember_qwen4exp_vision_provider PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

install(TARGETS ember_qwen4exp_vision_provider
  LIBRARY DESTINATION lib)
