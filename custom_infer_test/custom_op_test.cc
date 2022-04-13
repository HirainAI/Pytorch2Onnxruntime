/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "custom_op_infer.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
// #include "cuda_provider_factory.h"

// #include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
// #include <THC/THC.h>

typedef const char* PATH_TYPE;
#define TSTR(X) (X)
static constexpr PATH_TYPE MODEL_URI = TSTR("/root/tutorials-master/custom_infer_test/model.onnx");

template <typename T>
bool TestInference(Ort::Env& env, T model_uri,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<int32_t>& expected_values_y,
                   OrtCustomOpDomain* custom_op_domain_ptr) {
  Ort::SessionOptions session_options;
  std::cout << "Running simple inference with default provider" << std::endl;
  std::cout << "=================" << std::endl;
  // session_options.SetIntraOpNumThreads(1);
  // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // OrtCUDAProviderOptions cuda_options;
  // cuda_options.device_id = 0;
  // session_options.AppendExecutionProvider_CUDA(cuda_options);
  // void* cuda_compute_stream = nullptr;
  if (custom_op_domain_ptr) {
    std::cout << "=================000" << std::endl;
    session_options.Add(custom_op_domain_ptr);
    std::cout << session_options << std::endl;
  }
  // auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(cuda_compute_stream);
  // session_options.AppendExecutionProvider_CUDA(cuda_options);
  std::cout << "=================--" << std::endl;
  Ort::Session session(env, model_uri, session_options);
  std::cout << "=================" << std::endl;
  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> input_names;
  std::cout << "=================" << std::endl;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(info_cuda, const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  Ort::Value* input = &input_tensors[0];
  auto type_info1 = input->GetTensorTypeAndShapeInfo();
  std::cout << type_info1.GetShape() << std::endl;

  std::vector<Ort::Value> ort_outputs;
  std::cout << "session.run" << std::endl;
  ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);
  std::cout << "session.end" << std::endl;
  Ort::Value output_tensor{nullptr};
  output_tensor = Ort::Value::CreateTensor<int32_t>(info_cuda, const_cast<int32_t*>(expected_values_y.data()), expected_values_y.size(), expected_dims_y.data(), expected_dims_y.size());
  assert(ort_outputs.size() == 1);

  auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
  assert(type_info.GetShape() == expected_dims_y);
  size_t total_len = type_info.GetElementCount();
  assert(expected_values_y.size() == total_len);
  Ort::Value* out = &ort_outputs[0];
  int* f = out->GetTensorMutableData<int32_t>();

  for (size_t i = 0; i != total_len; ++i) {
    std::cout << f[i] << std::endl;
    std::cout << expected_values_y[i] << std::endl;
    // assert(expected_values_y[i] == f[i]);
  }

  return true;

}

int main(int argc, char** argv) {

  Ort::Env env_= Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");

  std::vector<Input> inputs(1);
  auto input = inputs.begin();
  input->name = "X1";
  input->dims = {1, 6, 3};
  input->values = { 0.7698f, 0.2795f, -1.8334f,  0.1331f, 0.4505f, -1.3810f, 0.6154f, 0.1577f, -0.8425f, -0.0553f, -1.0352f,  -0.1966f, -0.6620f, -1.0646f, -0.4216f, -1.2652f, -1.5929f, 0.1803f};
  std::cout << "=================" << std::endl;
  // input = std::next(input, 1);
  // input->name = "npoint";
  // input->dims = {1};
  // input->values = {3.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {1, 3};
  std::vector<int32_t> expected_values_y = { 0, 5, 3};

  GroupNormCustomOp custom_op;
  Ort::CustomOpDomain custom_op_domain("mydomain");
  custom_op_domain.Add(&custom_op);
  std::cout << "=================" << std::endl;
  return TestInference(env_, MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, custom_op_domain);
}
