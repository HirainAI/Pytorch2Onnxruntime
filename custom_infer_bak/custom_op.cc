/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "core/session/onnxruntime_c_api.h"

#include "sampling_gpu.h"
#include "custom_op_infer.h"
using namespace std;

void Compute(OrtKernelContext* context) {
  // Setup inputs
  // std::cout << "compute" << std::endl;
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const float* input_X1 = ort_.GetTensorData<float>(input_X);
  
  // const T* X_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_X));
  // const OrtValue* input_num_groups = ort_.KernelContext_GetInput(context, 1);
  // const T* num_groups = reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_num_groups));

  // Setup output
  // OrtTensorDimensions dimensions(ort_, input_X);    // *dimensions.data() = {1, 6, 3}, dimensions.size() = 3
  int B = 1;  //1
  int N = 6;  //6
  int npoint_ = 3;
  std::cout << "=================" << std::endl;
  int64_t dim_values[2] = {1, 3};
  const int64_t* dim_val = dim_values;
  size_t dim_count = 2;
  std::cout << "=================" << std::endl;
  // Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dim_val, dim_count);
  int* out = ort_.GetTensorMutableData<int32_t>(output);
  std::cout << *out << std::endl;
  // int* out = (int *)out_tmp;
  std::cout << "=================" << std::endl;
  float temp[6] = { 1e10, 1e10, 1e10, 1e10, 1e10, 1e10};
  float* temp_tensor = temp;
  // at::Tensor temp_tensor = torch::zeros({10}, torch::dtype(torch::kFloat32));

  // OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  // ort_.ReleaseTensorTypeAndShapeInfo(output_info);
  // Do computation
  std::cout << *temp_tensor << std::endl;
	int a = furthest_point_sampling_wrapper(B, N, npoint_, input_X1, temp_tensor, out);
  // for (size_t i = 0; i != 3; ++i) {
  //   std::cout << *out << std::endl;
  //   // assert(expected_values_y[i] == f[i]);
  // }
}
