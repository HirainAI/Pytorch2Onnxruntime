#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

// #include <torch/serialize/tensor.h>
// #include <ATen/cuda/CUDAContext.h>
#include<vector>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

int furthest_point_sampling_wrapper(int64_t b, int64_t n, int64_t m, 
    const float* points_tensor, float* temp_tensor, int64_t* idx_tensor);

void furthest_point_sampling_kernel_launcher(int64_t b, int64_t n, int64_t m, 
    const float* dataset, float* temp, int64_t* idxs);

// int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
//     at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

// void furthest_point_sampling_with_dist_kernel_launcher(int b, int n, int m,
//     const float *dataset, float *temp, int *idxs, cudaStream_t stream);

#endif
