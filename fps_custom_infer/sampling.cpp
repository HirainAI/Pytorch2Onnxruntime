/*
batch version of point sampling and gathering, modified from the original implementation of official PointNe
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


// #include <torch/serialize/tensor.h>
// #include <ATen/cuda/CUDAContext.h>
#include <vector>
// #include <THC/THC.h>
#include <limits>
#include <iostream>
#include "sampling_gpu.h"


// extern THCState *state;

// b: batch size
// n: num of points from where the points are sampled
// m: num of points to be sampled
// points_tensor: (b, n, 3) coordinates of input tensors
// tmp_tensor:
// output: (b, m) indices tensor of sampled points
int furthest_point_sampling_wrapper(int64_t input_points_num, int64_t output_points_num, const float* input_points, int64_t* idx) {

    // construct data ptr
    // const float *points = points_tensor;
    // float *temp = temp_tensor;
    // int *idx = idx_tensor;
  
    float temp[input_points_num] = {0};
    std::cout << "std::numeric_limits<float>::max() = " << std::numeric_limits<float>::max() << std::endl;
    for (size_t i = 0; i < input_points_num; i++)
    {
        temp[i]=std::numeric_limits<float>::max();
    }
    float* temp_tensor = temp;
    furthest_point_sampling_kernel_launcher(1, input_points_num, output_points_num, input_points, temp_tensor, idx);
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