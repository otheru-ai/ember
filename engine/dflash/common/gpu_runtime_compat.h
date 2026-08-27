#pragma once

// Minimal HIP compatibility for vendored code that retains cuda* spellings.
#include <hip/hip_runtime.h>

#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset
#define cudaMemGetInfo hipMemGetInfo
#define cudaSetDevice hipSetDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceReset hipDeviceReset
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFreeAsync hipFreeAsync
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaMallocAsync hipMallocAsync
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemsetAsync hipMemsetAsync
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDefault hipStreamDefault
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaErrorInvalidValue hipErrorInvalidValue
