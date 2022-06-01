/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>
#include <atomic>
#include <mutex>
#include <algorithm>
// #include <gtest/gtest.h>
#include <stdexcept>
#include <assert.h>
// #include "../cmake/external/onnxruntime-extensions/includes/onnxruntime/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/common/common.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include <experimental_onnxruntime_cxx_api.h>

// #include "/root/workspace/onnxruntime/onnxruntime/test/shared_lib/utils.h"
#include <cuda_runtime.h>
#include "/usr/local/cuda-11.2/targets/x86_64-linux/include/driver_types.h"
#include "/root/workspace/onnxruntime/onnxruntime/test/util/include/providers.h"
// #include "/root/workspace/onnxruntime/onnxruntime/test/util/include/test_allocator.h"
#include "/root/workspace/onnxruntime/onnxruntime/test/shared_lib/test_fixture.h"
// #include "/root/workspace/onnxruntime/onnxruntime/test/shared_lib/custom_op_utils.h"
#include <gsl/gsl>
// #include "providers.h"
#include "sampling_gpu.h"

struct Input {
  const char* name;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

struct MockedOrtAllocator : OrtAllocator {
  MockedOrtAllocator();
  ~MockedOrtAllocator();

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;
  size_t NumAllocations() const;

  void LeakCheck();

 private:
  MockedOrtAllocator(const MockedOrtAllocator&) = delete;
  MockedOrtAllocator& operator=(const MockedOrtAllocator&) = delete;

  std::atomic<size_t> memory_inuse{0};
  std::atomic<size_t> num_allocations{0};
  OrtMemoryInfo* cpu_memory_info;
};

// OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream = nullptr);


struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};
//定义算子内核

template <typename T>
struct FpsCustomKernel {
    private:
       Ort::CustomOpApi ort_;
        int64_t npoint_;
        void* compute_stream_;
    public:
        FpsCustomKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info, void* compute_stream)
      : ort_(ort_), compute_stream_(compute_stream_) {
        // npoint_ = ort_.KernelInfoGetAttribute<int64_t>(info, "npoint");
       }
        void Compute(OrtKernelContext* context);
        //     float temp[6] = { 1e10, 1e10, 1e10, 1e10, 1e10, 1e10};
        //     float* temp_tensor = temp;
        //    furthest_point_sampling_wrapper(npoint_, input_X1, temp_tensor, out);
        // }
};

// void furthest_point_sampling_wrapper(int npoint_, torch::Tensor input_X1, torch::Tensor out);
        //{
        //   const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
        //   const float* input_X1 = ort_.GetTensorData<float>(input_X);
        //   // const T* X_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_X));
        //   // const OrtValue* input_num_groups = ort_.KernelContext_GetInput(context, 1);
        //   // const T* num_groups = reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_num_groups));
        //   std::cout << "input_X1 "<<*input_X1 << std::endl;

        //   // Setup output
        //   // OrtTensorDimensions dimensions(ort_, input_X);    // *dimensions.data() = {1, 6, 3}, dimensions.size() = 3
        //   int B = 1;  //1
        //   int N = 6;  //6
        //   int npoint_ = 3;
        //   std::cout << "========compute=========" << std::endl;
        //   int64_t dim_values[2] = {1, 3};
        //   const int64_t* dim_val = dim_values;
        //   size_t dim_count = 2;
        //   // Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
        //   OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dim_val, dim_count);
        //   int* out = ort_.GetTensorMutableData<int32_t>(output);
        //   std::cout << "output "<<*out << std::endl;
        //   // int* out = (int *)out_tmp;
        //   float temp[6] = { 1e10, 1e10, 1e10, 1e10, 1e10, 1e10};
        //   float* temp_tensor = temp;
        //   // at::Tensor temp_tensor = torch::zeros({10}, torch::dtype(torch::kFloat32));

        //   // OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
        //   // ort_.ReleaseTensorTypeAndShapeInfo(output_info);
        //   // Do computation
        //   int a = furthest_point_sampling_wrapper(B, N, npoint_, input_X1, temp_tensor, out);
        // };

// template <typename T>

// 然后定义定制算子的各个操作，各个成员函数均已实现，其中 CreateKernel 会返回前面定义的算子核对象
struct FpsCustomOp : Ort::CustomOpBase<FpsCustomOp, FpsCustomKernel<float>> {
  // explicit FpsCustomOp(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new FpsCustomKernel<float>(api, info); };
  const char* GetName() const { return "FurthestPointSample_0"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // Both the inputs need to be necessarily of float type
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; };

//  private:
//   const char* provider_;
//   void* compute_stream_;
};

struct MyCustomKernel {
  MyCustomKernel(Ort::CustomOpApi ort, const OrtKernelInfo* /*info*/, void* compute_stream)
      : ort_(ort), compute_stream_(compute_stream) {
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  void* compute_stream_;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomKernel> {
  explicit MyCustomOp(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new MyCustomKernel(api, info, compute_stream_); };
  const char* GetName() const { return "Foo"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // Both the inputs need to be necessarily of float type
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_;
  void* compute_stream_;
};


