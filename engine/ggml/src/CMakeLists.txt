include(CheckCXXCompilerFlag)
include("../cmake/common.cmake")

add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (GGML_SANITIZE_THREAD)
    add_compile_options(-fsanitize=thread)
    link_libraries     (-fsanitize=thread)
endif()

if (GGML_SANITIZE_ADDRESS)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    link_libraries     (-fsanitize=address)
endif()

if (GGML_SANITIZE_UNDEFINED)
    add_compile_options(-fsanitize=undefined)
    link_libraries     (-fsanitize=undefined)
endif()

if (GGML_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS   -Werror)
        list(APPEND CXX_FLAGS -Werror)
    endif()
endif()

if (GGML_ALL_WARNINGS)
    list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
    list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                              -Werror=implicit-int -Werror=implicit-function-declaration)
    list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

    list(APPEND C_FLAGS   ${WARNING_FLAGS})
    list(APPEND CXX_FLAGS ${WARNING_FLAGS})

    ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

    add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
endif()

if (GGML_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (GGML_CCACHE AND NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(GGML_CCACHE_FOUND ccache)
    find_program(GGML_SCCACHE_FOUND sccache)

    if (GGML_CCACHE_FOUND OR GGML_SCCACHE_FOUND)
        if(GGML_CCACHE_FOUND)
            set(GGML_CCACHE_VARIANT ccache)
        else()
            set(GGML_CCACHE_VARIANT sccache)
        endif()
        # TODO: should not be set globally
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${GGML_CCACHE_VARIANT}")
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "${GGML_CCACHE_VARIANT} found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
endif()

# The parent configuration already rejects everything except Linux x86-64.
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
ggml_get_system_arch()
message(STATUS "GGML_SYSTEM_ARCH: ${GGML_SYSTEM_ARCH}")

if (GGML_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")
    add_link_options(-static)
endif()

if (GGML_GPROF)
    add_compile_options(-pg)
endif()

# ggml needs POSIX interfaces plus Linux CPU-affinity/memory extensions.
add_compile_definitions(_XOPEN_SOURCE=600 _GNU_SOURCE)

# ggml

add_library(ggml-base
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ../include/ggml-opt.h
            ../include/gguf.h
            ggml.c
            ggml.cpp
            ggml-alloc.c
            ggml-backend.cpp
            ggml-opt.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-quants.c
            ../rocmfp4/rocmfp4.c
            ../rocmfp4/rocmfp4.h
            ../rocmfpx/rocmfpx.c
            ../rocmfpx/rocmfpx.h
            ggml-quants.h
            gguf.cpp)

set_target_properties(ggml-base PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

target_include_directories(ggml-base PRIVATE .)

if (GGML_SCHED_NO_REALLOC)
    target_compile_definitions(ggml-base PUBLIC GGML_SCHED_NO_REALLOC)
endif()

add_library(ggml
            ggml-backend-reg.cpp)
add_library(ggml::ggml ALIAS ggml)

set_target_properties(ggml PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

target_link_libraries(ggml PUBLIC ggml-base)

function(ggml_add_backend_library backend)
    add_library(${backend} ${ARGN})
    target_link_libraries(ggml PUBLIC ${backend})
    target_link_libraries(${backend} PRIVATE ggml-base)
    target_include_directories(${backend} PRIVATE ..)

    if (${BUILD_SHARED_LIBS})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_BUILD)
        target_compile_definitions(${backend} PUBLIC  GGML_BACKEND_SHARED)
    endif()

    set_target_properties(${backend} PROPERTIES
        VERSION ${GGML_VERSION}
        SOVERSION ${GGML_VERSION_MAJOR})
endfunction()

add_subdirectory(ggml-cpu)
ggml_add_cpu_backend_variant_impl("")
target_compile_definitions(ggml PUBLIC GGML_USE_CPU)

add_subdirectory(ggml-hip)
target_compile_definitions(ggml PUBLIC GGML_USE_HIP)

foreach (target ggml-base ggml)
    target_include_directories(${target} PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<INSTALL_INTERFACE:include>)
    target_compile_features   (${target} PRIVATE c_std_11 cxx_std_17) # don't bump
endforeach()

target_link_libraries(ggml-base PRIVATE Threads::Threads)

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    target_link_libraries(ggml-base PRIVATE m)
endif()

if (BUILD_SHARED_LIBS)
    foreach (target ggml-base ggml)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(${target} PRIVATE GGML_BUILD)
        target_compile_definitions(${target} PUBLIC  GGML_SHARED)
    endforeach()
endif()
