#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

// #include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include<vector>
// #include <THC/THC.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

void furthest_point_sampling_kernel_launcher(int64_t b, int64_t n, int64_t m, 
    const float* dataset, float* temp, int64_t* idxs);

#endif
