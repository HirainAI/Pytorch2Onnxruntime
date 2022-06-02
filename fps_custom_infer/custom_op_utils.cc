/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
// #include "core/session/onnxruntime_c_api.h"

#include "sampling_gpu.h"
#include "custom_op_infer.h"

// template <typename T1, typename T2>
// void cuda_add(int64_t,  T1*, const T2*, cudaStream_t compute_stream);

using namespace std;

// template <typename T>
void FpsCustomKernel::Compute(OrtKernelContext* context){
        // Setup inputs
          const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
          const float* X = ort_.GetTensorData<float>(input_X);

          // Setup output
          OrtTensorDimensions dimensions(ort_, input_X);
          OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
          int64_t* out = ort_.GetTensorMutableData<int64_t>(output);

          OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
          int64_t size = ort_.GetTensorShapeElementCount(output_info);
          ort_.ReleaseTensorTypeAndShapeInfo(output_info);
          // Do computation
          furthest_point_sampling_wrapper(16384, npoint_, X, out);
          // std::cout << "output "<<*out << std::endl;

}
