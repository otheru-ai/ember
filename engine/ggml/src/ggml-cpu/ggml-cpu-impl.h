#pragma once

// GGML CPU internal header

#include "ggml.h"
#include "ggml-impl.h"

#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#include <stdbool.h>
#include <string.h> // memcpy
#include <math.h>   // fabsf

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#if defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#else

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#elif defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#include <immintrin.h>
#endif

// TODO: move to ggml-threading
void ggml_barrier(struct ggml_threadpool * tp);

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value);
int  ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value);

#ifdef __cplusplus
}
#endif
