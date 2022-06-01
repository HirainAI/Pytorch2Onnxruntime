/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "custom_op_infer.h"
#include <cassert>
#include <vector>
#include "/root/workspace/onnxruntime/onnxruntime/test/util/include/test_allocator.h"

// #include <THC/THC.h>
#define USE_CUDA 1
typedef const char* PATH_TYPE;
#define TSTR(X) (X)
static constexpr PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("/root/workspace/onnxruntime_inference_test/custom_infer_test_restruct/fps_model1.onnx");

std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

template<class T>
T vectorProduct(std::vector<T>& v)
{
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}
// template <typename T1, typename T2, typename T3>
// void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);

// static constexpr PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_1.onnx");
// extern unique_ptr<Ort::Env> ort_env;
OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream) {
  OrtCUDAProviderOptions cuda_options;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.has_user_compute_stream = cuda_compute_stream != nullptr ? 1 : 0;
  cuda_options.user_compute_stream = cuda_compute_stream;
  return cuda_options;
}

template <typename OutT>
void RunSession(Ort::MemoryInfo memoryinfo, OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<Input>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<OutT>& values_y,
                Ort::Value* output_tensor) {
  // 构建模型输入
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor<float>(memoryinfo, const_cast<float*>(inputs[i].values.data()),
                                        inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
        // Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(inputs[i].values.data()),
        //                         inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }
  // 运行 RUN
  std::vector<Ort::Value> ort_outputs;
  if (output_tensor)
    session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                       &output_name, output_tensor, 1);
  else {
    ort_outputs = session_object.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                     &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);
    output_tensor = &ort_outputs[0];
    std::cout<<"output_tensor: "<<output_tensor<<std::endl;
  }

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims_y);
  size_t total_len = type_info.GetElementCount();
  std::cout<<"total_len:" <<total_len<<std::endl;
  std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
      // print input shapes/dims
    output_node_dims = type_info.GetShape();
    printf("num_dims=%zu\n", output_node_dims.size());
    for (int j = 0; j < output_node_dims.size(); j++) printf(" dim %d=%jd\n",  j, output_node_dims[j]);
  
  ASSERT_EQ(values_y.size(), total_len);
  OutT* f = output_tensor->GetTensorMutableData<OutT>();
  // for (size_t i = 0; i != 4096; ++i) {
  //   std::cout<<"output: "<<f[i]<<std::endl;
  // }
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }

}


template <typename T>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   OrtCustomOpDomain* custom_op_domain_ptr,
                    void* cuda_compute_stream = nullptr) {
  Ort::SessionOptions session_options;
  std::cout << "Running simple inference with default provider" << std::endl;
    auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(cuda_compute_stream);
    session_options.AppendExecutionProvider_CUDA(cuda_options);
  if (custom_op_domain_ptr) {
    session_options.Add(custom_op_domain_ptr);
  }

  Ort::Session session(env, model_uri.c_str(), session_options);
  Ort::AllocatorWithDefaultOptions allocator;

  std::string inputName = session.GetInputName(0, allocator);
   std::vector<std::string> inputNames{inputName};
  Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  std:: vector<int64_t> inputDims = inputTensorInfo.GetShape();
  size_t inputTensorSize = vectorProduct(inputDims);
  std::cout << "inputTensorSize: " << inputTensorSize<< std::endl;
  std:: vector<Ort::Value> inputTensors;
  std::cout << "inputDims: " << inputDims[0]<< " "<<inputDims[1]<<" "<<inputDims[2]<< std::endl;

  // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // auto default_allocator = std::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);  // GPU memory

  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> input_names;
    std:: cout << "====name=====" <<  inputs[0].name<< std::endl;
    std:: cout << "inputs.size()" <<  inputs.size()<< std::endl;

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
    std::cout << "================" << *inputs[i].values.data() << std:: endl;
    std::cout << "================" << inputs[i].values.size() << std::endl;
    std:: cout << "================" << *inputs[i].dims.data() << std::endl;
    std:: cout << "================" << *(inputs[i].dims.data()+1) << std::endl;
    std:: cout << "================" << *(inputs[i].dims.data()+2) << std::endl;
    std::cout << "================" << inputs[i].dims.size() << std::endl;
  }
  std::vector<const char*> output_node_names = {"output"};
  std::vector<Ort::Value> ort_outputs;
  std::cout << "session.run" << std::endl;
  session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_node_names.data(), 1);
  std::cout << "session.end" << std::endl;


  Ort::Value output_tensor{nullptr};
  output_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(expected_values_y.data()), expected_values_y.size(), expected_dims_y.data(), expected_dims_y.size());
  // assert(ort_outputs.size() == 1);

  auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
  assert(type_info.GetShape() == expected_dims_y);
  size_t total_len = type_info.GetElementCount();
  assert(expected_values_y.size() == total_len);

  float* f = output_tensor.GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    assert(expected_values_y[i] == f[i]);
  }

}

// template <typename OutT>
// static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
//                           const std::vector<Input>& inputs,
//                           const char* output_name,
//                           const std::vector<int64_t>& expected_dims_y,
//                           const std::vector<OutT>& expected_values_y,
//                           int provider_type,
//                           OrtCustomOpDomain* custom_op_domain_ptr,
//                           const char* custom_op_library_filename,
//                           void** library_handle = nullptr,
//                           bool test_session_creation_only = false,
//                           void* cuda_compute_stream = nullptr) {
//   Ort::SessionOptions session_options;
//   if (provider_type == 1) {
// #ifdef USE_CUDA
//     std::cout << "Running simple inference with cuda provider" << std::endl;
//     std::cout<<"test cuda_compute_stream: "<<cuda_compute_stream<<std::endl;
//     auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(cuda_compute_stream);
//     session_options.AppendExecutionProvider_CUDA(cuda_options);
// #else
//     ORT_UNUSED_PARAMETER(cuda_compute_stream);
//     return;
// #endif
//   } else if (provider_type == 2) {
// #ifdef USE_DNNL
//     Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, 1));
//     std::cout << "Running simple inference with dnnl provider" << std::endl;
// #else
//     return;
// #endif
//   } else if (provider_type == 3) {
// #ifdef USE_NUPHAR
//     Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options,
//                                                                       /*allow_unaligned_buffers*/ 1, ""));
//     std::cout << "Running simple inference with nuphar provider" << std::endl;
// #else
//     return;
// #endif
//   } else {
//     std::cout << "Running simple inference with default provider" << std::endl;
//   }
//    std::cout << "Running session_options.Add" << std::endl;

//   if (custom_op_domain_ptr) {
//     session_options.Add(custom_op_domain_ptr);
//   }

//   if (custom_op_library_filename) {
//     Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(session_options,
//                                                              custom_op_library_filename, library_handle));
//     std::cout << "custom_op_library_filename" << custom_op_library_filename<<std::endl;
//   }
//   // if session creation passes, model loads fine
//   std::cout<<"model_uri.c_str()： "<<model_uri.c_str()<<std::endl;
//   Ort::Session session(env, model_uri.c_str(), session_options);
//   // caller wants to test running the model (not just loading the model)
//   if (!test_session_creation_only) {
//     // Now run
//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     auto default_allocator = make_unique<MockedOrtAllocator>();

//     //without preallocated output tensor
//     // RunSession<OutT>(default_allocator.get(),
//     //                  session,
//     //                  inputs,
//     //                  output_name,
//     //                  expected_dims_y,
//     //                  expected_values_y,
//     //                  nullptr);
//     //with preallocated output tensor
//     // Ort::Value value_y = Ort::Value::CreateTensor<float>(default_allocator.get(),
//     //                                                      expected_dims_y.data(), expected_dims_y.size());
//     // Ort::Value value_y{nullptr};
//     //test


//     Ort::Value value_y = Ort::Value::CreateTensor<float>(memory_info,
//                                                          expected_dims_y.data(), expected_dims_y.size());
//     std::cout<<"expected_dims_y.data(): "<<expected_dims_y.data()<<" expected_dims_y.size(): "<<expected_dims_y.size()<<std::endl;
//     //test it twice
//     std::cout<<"session run start"<<std::endl;
//     RunSession<OutT>(memory_info, default_allocator.get(),
//                        session,
//                        inputs,
//                        output_name,
//                        expected_dims_y,
//                        expected_values_y,
//                        &value_y);
  
//     std::cout<<"session run end"<<std::endl;
//     std::cout<<"expected_dims_y.data(): "<<expected_dims_y.data()<<" expected_dims_y.size(): "<<expected_dims_y.size()<<std::endl;
//   }
// }
// file path: onnxruntime/test/shared_lib/test_inference.cc

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
  std::cout<<"input.value[0]: "<<input.values[0]<<std::endl;
  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {1,4096};
  std::vector<float> expected_values_y(4096);
  std::generate(expected_values_y.begin(), expected_values_y.end(), [&] { return rand() % 255; }); 

  // 创建定制算子（MyCustomOp）
  cudaStream_t compute_stream = nullptr;    // 声明一个 cuda stream
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);  // 创建一个 cuda stream
  std::cout<<"compute_stream1: "<<compute_stream<<std::endl;
  FpsCustomOp custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};
  // FpsCustomOp custom_op_cpu{onnxruntime::kCpuExecutionProvider, nullptr};
  
  // 创建定制算子域（CustomOpDomain）
  Ort::CustomOpDomain custom_op_domain("mydomain");
  // 在定制算子域中添加定制算子
  custom_op_domain.Add(&custom_op);
  // custom_op_domain.Add(&custom_op);

  // 进入 TestInference
#ifdef USE_CUDA
  // TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "output", expected_dims_y, expected_values_y, 1,
  //                      custom_op_domain, nullptr, nullptr, false, compute_stream);
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "output", expected_dims_y, expected_values_y,  custom_op_domain,  compute_stream);
  cudaStreamDestroy(compute_stream);
#else
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr);
#endif
}
