
#include "sampling_gpu.h"

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
}

static auto registry = torch::RegisterOperators("mynamespace::furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper);
