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
#include <gtest/gtest.h>
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
// #include "/root/workspace/onnxruntime/onnxruntime/test/util/include/test_allocator.h"
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


// OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream = nullptr);


struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};
//定义算子内核

// template <typename T>
struct FpsCustomKernel {
        FpsCustomKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info, void* compute_stream): ort_(ort), compute_stream_(compute_stream) {
          npoint_ = ort_.KernelInfoGetAttribute<int64_t>(info, "npoint");
          std::cout<<"npoint: "<<npoint_<<std::endl;
       }
        void Compute(OrtKernelContext* context);
        private:
            Ort::CustomOpApi ort_;
            int64_t npoint_;
            void* compute_stream_;
};


// 然后定义定制算子的各个操作，各个成员函数均已实现，其中 CreateKernel 会返回前面定义的算子核对象
struct FpsCustomOp : Ort::CustomOpBase<FpsCustomOp, FpsCustomKernel> {
  explicit FpsCustomOp(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new FpsCustomKernel(api, info, compute_stream_); };
  const char* GetName() const { return "FurthestPointSample_0"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;};
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_;
  void* compute_stream_;
};