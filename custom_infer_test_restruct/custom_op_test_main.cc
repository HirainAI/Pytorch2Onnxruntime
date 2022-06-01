/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "custom_op_infer.h"
#include <cassert>
#include <vector>
// #include <THC/THC.h>
#define USE_CUDA 1
typedef const char* PATH_TYPE;
#define TSTR(X) (X)
static constexpr PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("/root/workspace/onnxruntime_inference_test/custom_infer_test_restruct/fps_model2.onnx");
using namespace std;
std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

template <typename T, size_t N>
constexpr size_t countof(T (&)[N]) { return N; }

OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream) {
  OrtCUDAProviderOptions cuda_options;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.has_user_compute_stream = cuda_compute_stream != nullptr ? 1 : 0;
  cuda_options.user_compute_stream = cuda_compute_stream;
  return cuda_options;
}

 int main(int argc, char** argv) {
  std::cout << "Running custom op inference" << std::endl;
 auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
 
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "input";
  input.dims = {1,16384,4};
  std::vector<float> v(16384 * 4);
   std::generate(v.begin(), v.end(), [&] { return rand() % 255; });
   input.values = v;
  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {1,4096};
  std::vector<float> expected_values_y(4096);
  std::generate(expected_values_y.begin(), expected_values_y.end(), [&] { return rand() % 255; }); 

  // 创建定制算子（MyCustomOp）
  cudaStream_t compute_stream = nullptr;    // 声明一个 cuda stream
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);  // 创建一个 cuda stream
  FpsCustomOp custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};

  
  // 创建定制算子域（CustomOpDomain）
  Ort::CustomOpDomain custom_op_domain("mydomain");
  // 在定制算子域中添加定制算子
  custom_op_domain.Add(&custom_op);
  // 进入 TestInference
   Ort::SessionOptions session_options;
    std::cout << "Running simple inference with cuda provider" << std::endl;
     std::basic_string<ORTCHAR_T> model_uri = CUSTOM_OP_MODEL_URI;
    auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(compute_stream);
    

    session_options.Add(custom_op_domain);
   Ort::Session session(*ort_env, model_uri.c_str(), session_options);
  // caller wants to test running the model (not just loading the model)

    // Now run
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto default_allocator = make_unique<MockedOrtAllocator>();
    //test it twice
    std::cout<<"session run start"<<std::endl;

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor<float>(memory_info,const_cast<float*>(inputs[i].values.data()),
                                        inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
        // Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(inputs[i].values.data()),
        //                         inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }
  // 运行 RUN
  // std::vector<Ort::Value> ort_outputs;
   const char* output_names[] = {"output"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));


    // ASSERT_EQ(ort_outputs.size(), 1u);
    Ort::Value* output_tensor = &ort_outputs[0];
    std::cout<<"output_tensor: "<<output_tensor<<std::endl;

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  // ASSERT_EQ(type_info.GetShape(), expected_dims_y);
  size_t total_len = type_info.GetElementCount();
  std::cout<<"total_len:" <<total_len<<std::endl;
  std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
      // print input shapes/dims
    output_node_dims = type_info.GetShape();
    printf("num_dims=%zu\n", output_node_dims.size());
    for (int j = 0; j < output_node_dims.size(); j++) printf(" dim %d=%jd\n",  j, output_node_dims[j]);
  
  // ASSERT_EQ(expected_values_y.size(), total_len);
  float* f = output_tensor->GetTensorMutableData<float>();
  // for (size_t i = 0; i != 4096; ++i) {
  //   std::cout<<"output: "<<f[i]<<std::endl;
  // }
  // for (size_t i = 0; i != total_len; ++i) {
  //   ASSERT_EQ(expected_values_y[i], f[i]);
  // }
  
  cudaStreamDestroy(compute_stream);

}
