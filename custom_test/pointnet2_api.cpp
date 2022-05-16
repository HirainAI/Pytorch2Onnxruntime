#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "sampling_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
    // m.def("furthest_point_sampling_with_dist_wrapper", &furthest_point_sampling_with_dist_wrapper, "furthest_point_sampling_with_dist_wrapper");

}
// static auto registry = torch::RegisterOperators("mynamespace::furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper);
