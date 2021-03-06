/*
batch version of point sampling and gathering, modified from the original implementation of official PointNe
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


// #include <torch/serialize/tensor.h>
// #include <ATen/cuda/CUDAContext.h>
#include <vector>
// #include <THC/THC.h>
#include <iostream>
#include "sampling_gpu.h"

// extern THCState *state;

// b: batch size
// n: num of points from where the points are sampled
// m: num of points to be sampled
// points_tensor: (b, n, 3) coordinates of input tensors
// tmp_tensor:
// output: (b, m) indices tensor of sampled points
int furthest_point_sampling_wrapper(int b, int n, int m, 
    const float* points, float* temp, int* idx) {

    // construct data ptr
    // const float *points = points_tensor;
    // float *temp = temp_tensor;
    // int *idx = idx_tensor;
    for (size_t i = 0; i != 3; ++i) {
    std::cout << *(temp+i) << std::endl;
    // assert(expected_values_y[i] == f[i]);
    }
    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
    for (size_t i = 0; i != 3; ++i) {
    std::cout << *(temp+i) << std::endl;
    // assert(expected_values_y[i] == f[i]);
    }
    return 1;
}

// int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
//     at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

//     const float *points = points_tensor.data<float>();
//     float *temp = temp_tensor.data<float>();
//     int *idx = idx_tensor.data<int>();

//     cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
//     furthest_point_sampling_with_dist_kernel_launcher(b, n, m, points, temp, idx, stream);
//     return 2;
// }