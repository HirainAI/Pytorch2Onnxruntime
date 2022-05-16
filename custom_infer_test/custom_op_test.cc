/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "custom_op_infer.h"
// #include "onnxruntime/core/session/onnxruntime_cxx_api.h"
// #include "onnxruntime/core/session/onnxruntime_c_api.h"
// #include "core/session/onnxruntime_cxx_api.h"
// #include "core/providers/cuda/cuda_provider_factory.h"
// #include "core/session/onnxruntime_c_api.h"
#include <cassert>
// #include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
// #include <torch/serialize/tensor.h>
// #include <core/session/onnxruntime_cxx_api.h>
// #include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
// #include <core/session/onnxruntime_c_api.h>
// #include <core/session/onnxruntime_cxx_api.h>
// #include "../cmake/external/onnxruntime-extensions/includes/onnxruntime/onnxruntime_c_api.h"
// #include "../cmake/external/onnxruntime-extensions/includes/onnxruntime/onnxruntime_cxx_api.h"
// #include <core/providers/tensorrt/tensorrt_provider_factory.h>
// #include <ATen/cuda/CUDAContext.h>
#include <vector>
// #include <THC/THC.h>
#define USE_CUDA 1
typedef const char* PATH_TYPE;
#define TSTR(X) (X)
static constexpr PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("/root/workspace/onnxruntime_inference_test/custom_infer_test/model.onnx");
using namespace std;

// template <typename T1, typename T2, typename T3>
// void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);

// static constexpr PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_1.onnx");
// extern unique_ptr<Ort::Env> ort_env;
auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream) {
  OrtCUDAProviderOptions cuda_options;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.has_user_compute_stream = cuda_compute_stream != nullptr ? 1 : 0;
  cuda_options.user_compute_stream = cuda_compute_stream;
  return cuda_options;
}

void FpsCustomKernel::Compute(OrtKernelContext* context){
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const float* input_X1 = ort_.GetTensorData<float>(input_X);
          // const T* X_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_X));
          // const OrtValue* input_num_groups = ort_.KernelContext_GetInput(context, 1);
          // const T* num_groups = reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_num_groups));
    std::cout << "input_X1 "<<*input_X1 << std::endl;

          // Setup output
          // OrtTensorDimensions dimensions(ort_, input_X);    // *dimensions.data() = {1, 6, 3}, dimensions.size() = 3
          int B = 1;  //1
          int N = 6;  //6
          int npoint_ = 3;
          std::cout << "========compute=========" << std::endl;
          int64_t dim_values[2] = {1, 3};
          const int64_t* dim_val = dim_values;
          size_t dim_count = 2;
          // Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
          OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dim_val, dim_count);
          int* out = ort_.GetTensorMutableData<int32_t>(output);
          std::cout << "output "<<*out << std::endl;
          // int* out = (int *)out_tmp;
          float temp[6] = { 1e10, 1e10, 1e10, 1e10, 1e10, 1e10};
          float* temp_tensor = temp;
          // at::Tensor temp_tensor = torch::zeros({10}, torch::dtype(torch::kFloat32));

          // OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
          // ort_.ReleaseTensorTypeAndShapeInfo(output_info);
          // Do computation
          int a = furthest_point_sampling_wrapper(B, N, npoint_, input_X1, temp_tensor, out);
}

// void MyCustomKernel::Compute(OrtKernelContext* context) {
//   // Setup inputs
//   const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
//   const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
//   const float* X = ort_.GetTensorData<float>(input_X);
//   const float* Y = ort_.GetTensorData<float>(input_Y);

//   // Setup output
//   OrtTensorDimensions dimensions(ort_, input_X);
//   OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
//   float* out = ort_.GetTensorMutableData<float>(output);

//   OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
//   int64_t size = ort_.GetTensorShapeElementCount(output_info);
//   ort_.ReleaseTensorTypeAndShapeInfo(output_info);

//   // Do computation
//   // Launch on stream 0 or user provided stream
//   cuda_add(size, out, X, Y, compute_stream_ == nullptr ? 0 : reinterpret_cast<cudaStream_t>(compute_stream_));
//   // If everything is setup correctly, custom op implementations need not have such synchronization logic.
//   // To make sure custom ops and ORT CUDA kernels are implicitly synchronized, create your session with a compute stream
//   // passed in via SessionOptions and use the same compute stream ti launch the custom op (as shown in this example)
//   // cudaStreamSynchronize(nullptr);

// }

template <typename OutT>
void RunSession(OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<Input>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<OutT>& values_y,
                Ort::Value* output_tensor) {
  // 构建模型输入
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor<float>(allocator->Info(allocator), const_cast<float*>(inputs[i].values.data()),
                                        inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }
  // 运行 RUN
  std::vector<Ort::Value> ort_outputs;
  if (output_tensor)
    session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                       &output_name, output_tensor, 1);
  else {
    ort_outputs = session_object.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                     &output_name, 1);
    // ASSERT_EQ(ort_outputs.size(), 1u);
    output_tensor = &ort_outputs[0];
    std::cout<<"output_tensor: "<<output_tensor<<std::endl;
  }

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  // ASSERT_EQ(type_info.GetShape(), dims_y);
  size_t total_len = type_info.GetElementCount();
  // ASSERT_EQ(values_y.size(), total_len);
  OutT* f = output_tensor->GetTensorMutableData<OutT>();
  // for (size_t i = 0; i != total_len; ++i) {
  //   ASSERT_EQ(values_y[i], f[i]);
  // }
}

template <typename OutT>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::vector<Input>& inputs,
                          const char* output_name,
                          const std::vector<int64_t>& expected_dims_y,
                          const std::vector<OutT>& expected_values_y,
                          int provider_type,
                          OrtCustomOpDomain* custom_op_domain_ptr,
                          const char* custom_op_library_filename,
                          void** library_handle = nullptr,
                          bool test_session_creation_only = false,
                          void* cuda_compute_stream = nullptr) {
  Ort::SessionOptions session_options;
  if (provider_type == 1) {
#ifdef USE_CUDA
    std::cout << "Running simple inference with cuda provider" << std::endl;
    auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(cuda_compute_stream);
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
    ORT_UNUSED_PARAMETER(cuda_compute_stream);
    return;
#endif
  } else if (provider_type == 2) {
#ifdef USE_DNNL
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, 1));
    std::cout << "Running simple inference with dnnl provider" << std::endl;
#else
    return;
#endif
  } else if (provider_type == 3) {
#ifdef USE_NUPHAR
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options,
                                                                      /*allow_unaligned_buffers*/ 1, ""));
    std::cout << "Running simple inference with nuphar provider" << std::endl;
#else
    return;
#endif
  } else {
    std::cout << "Running simple inference with default provider" << std::endl;
  }
   std::cout << "Running session_options.Add" << std::endl;

  if (custom_op_domain_ptr) {
    session_options.Add(custom_op_domain_ptr);
  }

  if (custom_op_library_filename) {
    Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(session_options,
                                                             custom_op_library_filename, library_handle));
    std::cout << "custom_op_library_filename" << custom_op_library_filename<<std::endl;
  }
  // if session creation passes, model loads fine
  std::cout<<"model_uri.c_str()： "<<model_uri.c_str()<<std::endl;
  Ort::Session session(env, model_uri.c_str(), session_options);
  // caller wants to test running the model (not just loading the model)
  if (!test_session_creation_only) {
    // Now run
    auto default_allocator = make_unique<MockedOrtAllocator>();

    //without preallocated output tensor
    RunSession<OutT>(default_allocator.get(),
                     session,
                     inputs,
                     output_name,
                     expected_dims_y,
                     expected_values_y,
                     nullptr);
    //with preallocated output tensor
    Ort::Value value_y = Ort::Value::CreateTensor<float>(default_allocator.get(),
                                                         expected_dims_y.data(), expected_dims_y.size());

    //test it twice
    for (int i = 0; i != 2; ++i){
            RunSession<OutT>(default_allocator.get(),
                       session,
                       inputs,
                       output_name,
                       expected_dims_y,
                       expected_values_y,
                       &value_y);
    }
  }
}

// file path: onnxruntime/test/shared_lib/test_inference.cc

 int main(int argc, char** argv) {
  std::cout << "Running custom op inference" << std::endl;

  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // 创建定制算子（MyCustomOp）
  cudaStream_t compute_stream = nullptr;    // 声明一个 cuda stream
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);  // 创建一个 cuda stream
  FpsCustomOp custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};

  
  // 创建定制算子域（CustomOpDomain）
  Ort::CustomOpDomain custom_op_domain("");
  // 在定制算子域中添加定制算子
  custom_op_domain.Add(&custom_op);
  // 进入 TestInference
#ifdef USE_CUDA
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 1,
                       custom_op_domain, nullptr, nullptr, false, compute_stream);
  cudaStreamDestroy(compute_stream);
#else
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr);
#endif
}

// template <typename T>
// static void TestInference(Ort::Env& env, T model_uri,
//                    const std::vector<Input>& inputs,
//                    const char* output_name,
//                    const std::vector<int64_t>& expected_dims_y,
//                    const std::vector<int32_t>& expected_values_y,
//                    OrtCustomOpDomain* custom_op_domain_ptr,
//                    void* cuda_compute_stream = nullptr) {
//   Ort::SessionOptions session_options;
//   std::cout << "Running simple inference with default provider" << std::endl;
//   std::cout << "=================" << std::endl;
//   session_options.SetIntraOpNumThreads(1);

//   //  OrtCUDAProviderOptions cuda_options{ 0 };
//   // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
//   // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

//   // OrtCUDAProviderOptions cuda_options;
//   // cuda_options.device_id = 0;
//   // cuda_options.arena_extend_strategy = 0;
//   // cuda_options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
//   // cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
//   // cuda_options.do_copy_in_default_stream = 1;
//   auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(cuda_compute_stream);
//   // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_options);
//   // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
// // //   // OrtCUDAProviderOptions cuda_options{0};
// //   session_options.AppendExecutionProvider_CUDA(cuda_options);
//   // cuda_options.device_id = 0;
//   session_options.AppendExecutionProvider_CUDA(cuda_options);
//   // session_options.AppendExecutionProvider_CUDA(cuda_options);
//   // void* cuda_compute_stream = nullptr;
//   if (custom_op_domain_ptr) {
//     std::cout << "============session option add=====" << std::endl;
//     session_options.Add(custom_op_domain_ptr);
//     std::cout <<  "============session option add complete=====" << std::endl;
//   }
//   // auto cuda_options = CreateDefaultOrtCudaProviderOptionsWcuda_optionsithCustomStream(cuda_compute_stream);
//   // session_options.AppendExecutionProvider_CUDA(cuda_options);
//   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
//   Ort::AllocatorWithDefaultOptions allocator;
//   std::cout <<  "============session set=====" << std::endl;
//   Ort::Session session(env, model_uri.c_str(), session_options);
//   std::cout <<  "============session set complete=====" << std::endl;
//   Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
//   // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//   std::vector<Ort::Value> input_tensors;
//   std::vector<const char*> input_names;
//   for (size_t i = 0; i < inputs.size(); i++) {
//     input_names.emplace_back(inputs[i].name);
//     input_tensors.emplace_back(Ort::Value::CreateTensor<float>(info_cuda, const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
//   }

//   Ort::Value* input = &input_tensors[0];
//   auto type_info1 = input->GetTensorTypeAndShapeInfo();
//   // std::cout << type_info1.GetShape() << std::endl;

//   std::vector<Ort::Value> ort_outputs;
//   std::cout << "session.run" << std::endl;
//   ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);
//   std::cout << "session.end" << std::endl;
//   Ort::Value output_tensor{nullptr};
//   output_tensor = Ort::Value::CreateTensor<int32_t>(info_cuda, const_cast<int32_t*>(expected_values_y.data()), expected_values_y.size(), expected_dims_y.data(), expected_dims_y.size());
//   assert(ort_outputs.size() == 1);

//   auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
//   assert(type_info.GetShape() == expected_dims_y);
//   size_t total_len = type_info.GetElementCount();
//   assert(expected_values_y.size() == total_len);
//   Ort::Value* out = &ort_outputs[0];
//   int* f = out->GetTensorMutableData<int32_t>();

//   // for (size_t i = 0; i != total_len; ++i) {
//   //   std::cout << f[i] << std::endl;
//   //   std::cout << expected_values_y[i] << std::endl;
//   //   // assert(expected_values_y[i] == f[i]);
//   // }

// }

// int main(int argc, char** argv) {
//   std::string MODEL_URI = "/root/workspace/onnxruntime_inference_test/custom_infer_test/model.onnx";
//   Ort::Env env_= Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");

//   std::vector<Input> inputs(1);
//   auto input = inputs.begin();
//   input->name = "X1";
//   input->dims = {1, 6, 3};
//   input->values = { 0.7698f, 0.2795f, -1.8334f,  0.1331f, 0.4505f, -1.3810f, 0.6154f, 0.1577f, -0.8425f, -0.0553f, -1.0352f,  -0.1966f, -0.6620f, -1.0646f, -0.4216f, -1.2652f, -1.5929f, 0.1803f};
//   std::cout << "========flag1=========" << std::endl;
//   // input = std::next(input, 1);
//   // input->name = "npoint";
//   // input->dims = {1};
//   // input->values = {3.0f};

//   // prepare expected inputs and outputs
//   std::vector<int64_t> expected_dims_y = {1, 3};
//   std::vector<int32_t> expected_values_y = { 0, 5, 3};
//   // MyCustomOp custom_op_cpu{onnxruntime::kCpuExecutionProvider, nullptr};
//   cudaStream_t compute_stream = nullptr;    // 声明一个 cuda stream
//   cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
//   MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider, compute_stream};
//   Ort::CustomOpDomain custom_op_domain("mydomain");
//   custom_op_domain.Add(&custom_op);
//   // custom_op_domain.Add(&custom_op_cpu);
//   std::cout << "========flag2=========" << std::endl;
//   TestInference(env_, MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, custom_op_domain,compute_stream);
//   cudaStreamDestroy(compute_stream);
// }
