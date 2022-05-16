/*
batch version of point sampling and gathering, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


// #include <torch/serialize/tensor.h>

#include "sampling_gpu.h"

// extern THCState *state;

// b: batch size
// n: num of points from where the points are sampled
// m: num of points to be sampled
// points_tensor: (b, n, 3) coordinates of input tensors
// tmp_tensor:
// output: (b, m) indices tensor of sampled points
void furthest_point_sampling_wrapper(int64_t b, int64_t n, int64_t m, 
    torch::Tensor points_tensor, torch::Tensor temp_tensor, torch::Tensor idx_tensor) {

    // construct data ptr
    const float *points = points_tensor.data_ptr<float>();
    float *temp = temp_tensor.data_ptr<float>();
    int64_t *idx = idx_tensor.data_ptr<int64_t>();
    
    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
    m.def("furthest_point_sampling_kernel_launcher", &furthest_point_sampling_kernel_launcher, "furthest_point_sampling_kernel_launcher");
}

// TORCH_LIBRARY(fps, m) {
//     m.def("furthest_point_sampling_wrapper", furthest_point_sampling_wrapper);
//     // m.def("furthest_point_sampling_kernel_launcher", furthest_point_sampling_kernel_launcher);
// }

static auto registry = torch::RegisterOperators("mynamespace::furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper);

// int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
//     at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

//     const float *points = points_tensor.data<float>();
//     float *temp = temp_tensor.data<float>();
//     int *idx = idx_tensor.data<int>();

//     cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
//     furthest_point_sampling_with_dist_kernel_launcher(b, n, m, points, temp, idx, stream);
//     return 2;
// }