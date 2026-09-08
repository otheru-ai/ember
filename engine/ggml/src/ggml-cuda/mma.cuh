#pragma once
// This file contains primitives that expose the tensor core PTX instructions for CUDA code.
// The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout.
// The documentation for the PTX instructions can be found under:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
//
// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape M x K.
// B is a column-major matrix with shape K x N.
// C is a column-major matrix with shape M x N.
// A, B, and C are represented using the same fundamental data type: a row-major matrix with I rows and J columns.
// Note that J is measured in physical 32 bit elements instead of logical elements.
// The methods get_i and get_j can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the PTX documentation, all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.
// The API in this file also assumes that the pointers for load_generic are aligned to 16 bytes, unaligned pointers are considered undefined behavior.

#include "common.cuh"

// On Volta each warp is doing 4 8x8 mma operations in parallel.
// The basic memory layout for a 32x8 output tile is to stack 4 input tiles in I direction and to mirror the B tile.
// However, the i indices in this file are by default permuted to simplify the index calculations.
// #define GGML_CUDA_MMA_NO_VOLTA_PERM

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    // Imagine transposing row-major matrix to column-major matrix.
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

static __device__ __forceinline__ half2 ggml_cuda_movmatrix(const half2 x) {
    half2 ret;
    *((int *) &ret) = ggml_cuda_movmatrix(*((const int *) &x));
    return ret;
}

namespace ggml_cuda_mma {

    // Some architectures like Volta or CDNA3 perform multiple matrix multiplications per warp in parallel,
    //     effectively the warp is being split into subgroups of threads that each perform a single mma instruction.
    // In those cases the data can be split in different ways across the warp.
    enum data_layout {
        // By default the data uses the I direction as its major dimension and the J direction as its minor dimension.
        // For the A/C matrices this means I major == row major, J major == column major.
        // For the B matrix this means I major == column major, J major == row major.
        // MIRRORED == Each data value is held exactly once per thread subgroup.
        DATA_LAYOUT_I_MAJOR           =  0, // Turing+, CDNA, and matrix A&B for RDNA3.
        DATA_LAYOUT_J_MAJOR           = 10, // Matrix C for CDNA and integer/float matrix C for RDNA3.
        DATA_LAYOUT_I_MAJOR_MIRRORED  = 20, // Volta, matrix A&B for RDNA3.
        DATA_LAYOUT_J_MAJOR_MIRRORED  = 30,
    };
    // Implemented mma combinations are:
    //   - (I_MAJOR, I_MAJOR)          -> I_MAJOR
    //   - (I_MAJOR, I_MAJOR_MIRRORED) -> I_MAJOR
    //   - (I_MAJOR, J_MAJOR_MIRRORED) -> I_MAJOR

    static constexpr bool is_i_major(const data_layout dl) {
        return dl == DATA_LAYOUT_I_MAJOR ||
               dl == DATA_LAYOUT_I_MAJOR_MIRRORED;
    }

    static constexpr __device__ data_layout get_input_data_layout() {
#if defined(RDNA3)
        return DATA_LAYOUT_I_MAJOR_MIRRORED;
#else
        return DATA_LAYOUT_I_MAJOR;
#endif // defined(RDNA3)
    }

    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if   defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8) return true;
            if (I == 16 && J == 4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 16) {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                    // matrix C
                    return 2 * l + (threadIdx.x / 16);
                } else {
                    // matrix A&B
                    return l;
                }
            } else if constexpr (I == 16 && J == 8) {
                return ne * (threadIdx.x / 16) + l;
            } else if constexpr (I == 16 && J == 4) {
                return ne * (threadIdx.x / 16) + l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#else
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 16) {
                return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_i(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_j(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#endif // defined(GGML_USE_HIP)
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if   defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = I * J / 32;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 16 && J == 8) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 8) {
                return ne * (threadIdx.x / 16) + l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#else
        static constexpr int ne = I * J / WARP_SIZE;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 4) {
                return (l * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((l % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return ((l / 4) * 16) + ((l % 2) * 8) + (threadIdx.x / 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 32 && J == 8) {
                return ((l & 2) * 2) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#endif // defined(AMD_WMMA_AVAILABLE)
    };

    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::ne;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }
#else
        static constexpr int ne = I * J / WARP_SIZE;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  4) return true;
            if (I == 16 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 4) {
                return (l * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((l % 2) * 8) + (threadIdx.x / 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#endif  // defined(AMD_WMMA_AVAILABLE)
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR;

        static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;

        // RDNA3
        static constexpr int         ne = I * J / 32 * 2;

        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8)  return true;
            if (I == 16 && J == 4)  return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int /*l*/) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (supported()) {
                return l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;
#if defined(RDNA3)
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
#else // Volta
        static constexpr int         ne = I * J / (WARP_SIZE/4);

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int /*l*/) {
            if constexpr (I == 8 && J == 4) {
                return ((threadIdx.x / 16) * 4) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#endif // defined(RDNA3)
    };

    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR_MIRRORED;
        static constexpr int         ne = I * J / (WARP_SIZE/4);

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return ((threadIdx.x / 16) * 2) + (l % 2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

#if defined(AMD_WMMA_AVAILABLE)
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
        NO_DEVICE_CODE;
        return tile<8, 8, half2>{};
    }
#else // Volta
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 4) {
            ret.x[l0/2 + 0] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
            ret.x[l0/2 + 1] = make_half2(tile_float.x[l0 + 2], tile_float.x[l0 + 3]);

            // On Volta FP16 and FP32 tiles have a different memory layout,
            //     for the conversion threads with an offset of 2 need to exchange half their values:
            ret.x[l0/2 + (((threadIdx.x % 4) / 2) ^ 1)] = __shfl_xor_sync(
                0xFFFFFFFF, ret.x[l0/2 + (((threadIdx.x % 4) / 2) ^ 1)], 2, WARP_SIZE);
        }
        return ret;
    }
#endif // defined(TURING_MMA_AVAILABLE)

    static __device__ __forceinline__ void make_identity_mat(tile<16, 8, half2> & t) {
        GGML_UNUSED_VARS(t);
        NO_DEVICE_CODE;
    }

    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#if   defined(AMD_WMMA_AVAILABLE)
        // All wmma layout has contiguous data when i-major.
        if constexpr (is_i_major(dl)) {
            // the data must be aligned to 16 bytes when bigger than ggml_cuda_get_max_cpy_bytes()
            constexpr int aligned_copy_bytes = ggml_cuda_get_max_cpy_bytes();
            if constexpr (sizeof(t.x) > aligned_copy_bytes) {
                static_assert(sizeof(t.x) % aligned_copy_bytes == 0, "bad type size");
                constexpr int aligned_copy_count = sizeof(t.x)/aligned_copy_bytes;
#pragma unroll
                for (int i = 0; i < aligned_copy_count; ++i) {
                    ggml_cuda_memcpy_1<aligned_copy_bytes>(t.x + t.ne/aligned_copy_count*i, xs0 + t.get_i(0) * stride + t.get_j(t.ne/aligned_copy_count*i));
                }
            } else {
                ggml_cuda_memcpy_1<sizeof(t.x)>(t.x, xs0 + t.get_i(0) * stride + t.get_j(0));
            }
        } else {
#pragma unroll
            for (int l = 0; l < t.ne; ++l) {
                t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
            }
        }
#else
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
#endif // defined(AMD_MFMA_AVAILABLE)
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        load_generic(t, xs0, stride);
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#if   defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
        static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
        static_assert(sizeof(t.x) == 16, "bad ne");
        ggml_cuda_memcpy_1<8>(t.x + 0, xs0 + t.get_i(0)*stride + 0);
        ggml_cuda_memcpy_1<8>(t.x + 2, xs0 + t.get_i(0)*stride + 2);
#else
        static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad data layout");
        static_assert(sizeof(t.x) == 8, "bad ne");
        ggml_cuda_memcpy_1<8>(t.x, xs0 + t.get_i(0)*stride + t.get_j(0));
#endif
#else
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        load_generic(t, xs0, stride);
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
        ggml_cuda_memcpy_1<4*sizeof(half2)>(t.x, xs0 + t.get_i(0)*stride);
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l0 = 0; l0 < t.ne; l0 += 2) {
            ggml_cuda_memcpy_1<2*sizeof(half2)>(t.x + l0, xs0 + t.get_i(l0)*stride + t.get_j(l0));
        }
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<32, 4, half2> & t, const half2 * __restrict__ xs0, const int stride) {
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix_trans(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 4, half2> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, half2> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
#if   defined(AMD_WMMA_AVAILABLE)
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 8, float, dl_d> & D, const tile<16, 8, float, dl_ab> & A, const tile<8, 8, float, dl_ab> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, float, dl_ab> & A, const tile<16, 8, float, dl_ab> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma_block_scaled(tile<16, 8, float> &     D,
                                                            const tile<16, 8, int> & A,
                                                            const tile<8, 8, int> &  B,
                                                            uint32_t                 a_scale,
                                                            uint32_t                 b_scale) {
        GGML_UNUSED_VARS(D, A, B, a_scale, b_scale);
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, nv_bfloat162> & A, const tile<8, 8, nv_bfloat162> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, half2, dl_ab> & A, const tile<16, 8, half2, dl_ab> & B) {
#if   defined(AMD_WMMA_AVAILABLE)
        using halfx16_t = __attribute__((ext_vector_type(16))) _Float16;
        using floatx8_t = __attribute__((ext_vector_type(8))) float;
        floatx8_t& acc_frag = reinterpret_cast<floatx8_t&>(D.x[0]);
        const halfx16_t& a_frag = reinterpret_cast<const halfx16_t&>(A.x[0]);
        const halfx16_t& b_frag = reinterpret_cast<const halfx16_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc_frag);
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, nv_bfloat162, dl_ab> & A, const tile<16, 8, nv_bfloat162, dl_ab> & B) {
#if defined(AMD_WMMA_AVAILABLE)
        using bf16x16_t = __attribute__((ext_vector_type(16))) __bf16;
        using floatx8_t = __attribute__((ext_vector_type(8))) float;
        floatx8_t& acc_frag = reinterpret_cast<floatx8_t&>(D.x[0]);
        const bf16x16_t& a_frag = reinterpret_cast<const bf16x16_t&>(A.x[0]);
        const bf16x16_t& b_frag = reinterpret_cast<const bf16x16_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc_frag);
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE)
    }

    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 8, int, dl_ab> & A, const tile<16, 8, int, dl_ab> & B) {
#if   defined(AMD_WMMA_AVAILABLE)

        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        int32x8_t * acc = (int32x8_t *) D.x;

        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * a_vec = (int32x4_t *) A.x;
        int32x4_t * b_vec = (int32x4_t *) B.x;

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[0],
            true,
            b_vec[0],
            acc[0],
            true
        );

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[1],
            true,
            b_vec[1],
            acc[0],
            true
        );

#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma(
            tile<32, 32, int> & D, const tile<32, 4, int> & A, const tile<32, 4, int> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    template <typename T1, typename T2, int J, int K>
    static __device__ __forceinline__ void mma(
            tile<32, J, T1> & D, const tile<32, K, T2> & A, const tile<J, K, T2> & B) {
        tile      <16, J, T1> * D16 = reinterpret_cast<      tile<16, J, T1> *>(&D);
        const tile<16, K, T2> * A16 = reinterpret_cast<const tile<16, K, T2> *>(&A);
        mma(D16[0], A16[0], B);
        mma(D16[1], A16[1], B);
    }

    static __device__ __forceinline__ void mma(
            tile<32, 8, float> & D, const tile<32, 4, half2> & A, const tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<32, 4, half2> & D, const tile<32, 4, half2> & A, const tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & B) {
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 4, int, dl_ab> & A, const tile<16, 4, int, dl_ab> & B) {
#if defined(AMD_WMMA_AVAILABLE)
        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        int32x8_t * acc = (int32x8_t *) D.x;
        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * a_vec = (int32x4_t *) A.x;
        int32x4_t * b_vec = (int32x4_t *) B.x;

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[0],
            true,
            b_vec[0],
            acc[0],
            false
        );
#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // AMD_WMMA_AVAILABLE
    }

    template <bool b_signed = true, data_layout dl_d = DATA_LAYOUT_I_MAJOR, data_layout dl_ab = DATA_LAYOUT_I_MAJOR>
    static __device__ __forceinline__ void mma_iu4(
            tile<16, 16, int, dl_d> & D, const tile<16, 4, int, dl_ab> & A, const tile<16, 4, int, dl_ab> & B) {
#if defined(AMD_WMMA_AVAILABLE) && defined(__gfx1151__)
        // AMD machine-readable ISA, amdgpu_isa_rdna3_5.xml (internal release
        // 2026-02-20; archive-entry timestamp 2026-08-04):
        // V_WMMA_I32_16X16X16_IU4 consumes one 64-bit A and B fragment per
        // lane, produces one 256-bit I32 accumulator, and requires two A/B
        // copies across a wave32.  DATA_LAYOUT_I_MAJOR_MIRRORED supplies the
        // copies through lane%16; the two calls below cover the tile's two
        // consecutive K=16 groups.  Do not use the gfx12 non-replicated map.
        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
        static_assert(sizeof(int32x2_t) == 8, "gfx1151 IU4 WMMA operand must be 64 bits");
        static_assert(sizeof(int32x8_t) == 32, "gfx1151 IU4 WMMA accumulator must be 256 bits");
        static_assert(dl_ab == DATA_LAYOUT_I_MAJOR_MIRRORED,
                      "gfx1151 IU4 WMMA inputs must use the RDNA3.5 replicated layout");
        int32x8_t * acc = (int32x8_t *) D.x;
        int32x2_t * av = (int32x2_t *) A.x;
        int32x2_t * bv = (int32x2_t *) B.x;
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, av[0], b_signed, bv[0], acc[0], false);
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, av[1], b_signed, bv[1], acc[0], false);
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif
    }
}
