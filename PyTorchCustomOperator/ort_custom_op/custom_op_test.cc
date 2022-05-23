/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "custom_op.h"
#include "onnxruntime_cxx_api.h"

typedef const char* PATH_TYPE;
#define TSTR(X) (X)
static constexpr PATH_TYPE MODEL_URI = TSTR("/root/workspace/onnxruntime_inference_test/custom_infer_test/model.onnx");

template <typename T>
bool TestInference(Ort::Env& env, T model_uri,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   OrtCustomOpDomain* custom_op_domain_ptr) {
  Ort::SessionOptions session_options;
  std::cout << "Running simple inference with default provider" << std::endl;

  if (custom_op_domain_ptr) {
    session_options.Add(custom_op_domain_ptr);
  }

  Ort::Session session(env, model_uri, session_options);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> input_names;

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
    cout << "================" << *inputs[i].values.data() << endl;
    cout << "================" << inputs[i].values.size() << endl;
    cout << "================" << *inputs[i].dims.data() << endl;
    cout << "================" << inputs[i].dims.size() << endl;
  }

  std::vector<Ort::Value> ort_outputs;
  std::cout << "session.run" << std::endl;
  ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);
  std::cout << "session.end" << std::endl;


  Ort::Value output_tensor{nullptr};
  output_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(expected_values_y.data()), expected_values_y.size(), expected_dims_y.data(), expected_dims_y.size());
  assert(ort_outputs.size() == 1);

  auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
  assert(type_info.GetShape() == expected_dims_y);
  size_t total_len = type_info.GetElementCount();
  assert(expected_values_y.size() == total_len);

  float* f = output_tensor.GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    assert(expected_values_y[i] == f[i]);
  }

  return true;

}

int main(int argc, char** argv) {

  Ort::Env env_= Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");

  std::vector<Input> inputs(4);
  auto input = inputs.begin();
  input->name = "X";
  input->dims = {3, 2, 1, 2};
  input->values = { 1.5410f, -0.2934f, -2.1788f,  0.5684f, -1.0845f, -1.3986f , 0.4033f,  0.8380f, -0.7193f, -0.4033f ,-0.5966f,  0.1820f};

  input = std::next(input, 1);
  input->name = "num_groups";
  input->dims = {1};
  input->values = {2.f};

  input = std::next(input, 1);
  input->name = "scale";
  input->dims = {2};
  input->values = {2.0f, 1.0f};

  input = std::next(input, 1);
  input->name = "bias";
  input->dims = {2};
  input->values = {1.f, 0.f};
//  std::vector<Input> inputs(2);
//   auto input = inputs.begin();
//   input->name = "X1";
//   input->dims = {1,6,3};
//   input->values = { 1.5410f, -0.2934f, -2.1788f,  \
//                                0.5684f, -1.0845f, -1.3986f ,\
//                                0.4033f,  0.8380f, -0.7193f,\
//                                -0.4033f ,-0.5966f,  0.1820f,\
//                                -0.1863,  0.9658,  0.8852,\
//                                -0.8782, -2.5432, -0.5395};

//   input = std::next(input, 1);
//   input->name = "npoints";
//   input->dims = {1};
//   input->values = {6};
    std::vector<int64_t> expected_dims_y = {1};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
  // std::cout<<*
  // prepare expected inputs and outputs
  // prepare expected inputs and outputs
  // std::vector<int64_t> expected_dims_y = {3, 2, 1, 2};
  // std::vector<float> expected_values_y = { 3.0000f, -1.0000f, -1.0000f,  1.0000f, 2.9996f, -0.9996f, -0.9999f,  0.9999f,  -0.9996f,  2.9996f, -1.0000f,  1.0000f};

  GroupNormCustomOp custom_op;
  // Ort::custom_op_domain("mydomain");
  // custom_op_domain.Add(&custom_op);
   // 创建定制算子域（CustomOpDomain）
  Ort::CustomOpDomain custom_op_domain("mydomain");
  // 在定制算子域中添加定制算子
  custom_op_domain.Add(&custom_op);

  return TestInference(env_, MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, custom_op_domain);
}
