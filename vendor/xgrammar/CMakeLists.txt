# Static xgrammar core for ember. Written here rather than reusing upstream's
# CMakeLists because upstream unconditionally descends into cpp/tvm_ffi (its
# Python bindings), which requires a tvm_ffi CMake package and a Python
# toolchain that ember has no use for. Upstream already excludes those same
# sources from the core library, so dropping the directory costs nothing.
#
# See VENDOR.md for the pinned upstream commit and exactly what was omitted.

add_library(xgrammar STATIC
  cpp/compiled_grammar.cc
  cpp/config.cc
  cpp/earley_parser.cc
  cpp/fsm.cc
  cpp/fsm_builder.cc
  cpp/grammar.cc
  cpp/grammar_builder.cc
  cpp/grammar_compiler.cc
  cpp/grammar_functor.cc
  cpp/grammar_matcher.cc
  cpp/grammar_parser.cc
  cpp/grammar_printer.cc
  cpp/json_schema_converter.cc
  cpp/json_schema_converter_ext.cc
  cpp/regex_converter.cc
  cpp/structural_tag.cc
  cpp/testing.cc
  cpp/tokenizer_info.cc
  cpp/support/logging.cc
  cpp/support/recursion_guard.cc
)

target_include_directories(xgrammar
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/picojson
)

# cpp/support/cpptrace.h is a shim that degrades to no-ops when this is 0, which
# is upstream's own default. Keeping it off means the cpptrace submodule is not
# vendored and nothing third-party has to be compiled.
target_compile_definitions(xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)

target_compile_features(xgrammar PUBLIC cxx_std_17)

# Vendored third-party code is not held to ember's warning bar, and upstream
# builds with -Werror against a different compiler: clang flags std::result_of
# as deprecated in C++17 where GCC does not.
target_compile_options(xgrammar PRIVATE -w)

find_package(Threads REQUIRED)
target_link_libraries(xgrammar PUBLIC Threads::Threads)
