#pragma once
#include "cuda_runtime.h"
#include <stdio.h>

const int block_size = 256;
#define EPSILON (1e-6f)
#define PI (3.14159265358979323846f)
#define CUDA_CALL(x)                                                           \
  do {                                                                         \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA Error at %s:%d\t Error code = %d\n", __FILE__, __LINE__,    \
             x);                                                               \
    }                                                                          \
  } while (0)
// #define CUDA_CALL(x) do { x ;} while(0)
#define CHECK_KERNEL()                                                         \
  ;                                                                            \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err)                                                                   \
      printf("CUDA Error at %s:%d:\t%s\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(err));                                         \
  }
#define MAX_A (1000.0f)

namespace ThrustHelper {
template <typename T> struct plus {
  T _a;
  plus(const T a) : _a(a) {}
  __host__ __device__ T operator()(const T &lhs) const { return lhs + _a; }
};

template <typename T> struct abs_plus {
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {
    return abs(lhs) + abs(rhs);
  }
};
} // namespace ThrustHelper
