cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++14)
find_package(CUDA REQUIRED)
set(TEST_SOURCE sampling.cpp sampling_gpu.cu cuda_ops.cu custom_op_test.cc test_allocator.cc)
 
set(HEADER custom_op_infer.h sampling_gpu.h)

set(ONNXRUNTIME_ROOT_PATH /root/workspace/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/
                             /root/workspace/pytorch/third_party/
                           )
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)
# include_directories(
#                                     ${ONNXRUNTIME_ROOT_PATH}/include/
#                                     ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
#                                     ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/
#                                     # ${ONNXRUNTIME_ROOT_PATH}/cmake/external/onnxruntime-extensions/includes/onnxruntime/
#                                 )
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
# include_directories(/root/tutorials-master/custom_infer_test/)
# include_directories("/workspace/onnxruntime/cmake/external/onnxruntime-extensions/includes/onnxruntime/")
# include_directories(/opt/anaconda3/envs/3dssd_pytorch/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/)
# include_directories(/opt/anaconda3/envs/3dssd_pytorch/lib/python3.7/site-packages/torch/include/)
include_directories("/usr/local/cuda/include/")
cuda_add_executable(customop  ${HEADER} ${TEST_SOURCE})
#Include path to header files for Custom Op
# include_directories(/tmp/onnxInstall/build/native/include/)
message(STATUS ONNXRUNTIME_INCLUDE_DIRS)
link_libraries(/root/workspace/onnxruntime_inference_test/custom_infer_test/fps.cpython-37m-x86_64-linux-gnu.so)



#Include path to header files for Custom Op Test
#include_directories(/tmp/onnxInstall/runtimes/linux-x64/native)

#Linking dependencies for Custom Op
# find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ONNXRUNTIME_LIB)

target_link_libraries(customop 
                                ${ONNXRUNTIME_LIB}
                                )