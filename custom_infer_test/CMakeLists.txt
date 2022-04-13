cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++14)
find_package(CUDA REQUIRED)
set(TEST_SOURCE sampling.cpp sampling_gpu.cu custom_op_test.cc)
 
set(HEADER custom_op_infer.h)
set(SOURCE custom_op_infer.h)

cuda_add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})
#Include path to header files for Custom Op
include_directories(/tmp/onnxInstall/build/native/include/)
include_directories(/root/tutorials-master/custom_infer_test/)
include_directories(/opt/anaconda3/envs/3dssd_pytorch/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/)
include_directories(/opt/anaconda3/envs/3dssd_pytorch/lib/python3.7/site-packages/torch/include/)
include_directories(/usr/local/cuda/include/)
link_libraries(/root/tutorials-master/custom_infer_test/fps.cpython-37m-x86_64-linux-gnu.so)



#Include path to header files for Custom Op Test
#include_directories(/tmp/onnxInstall/runtimes/linux-x64/native)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /tmp/onnxInstall/runtimes/linux-x64/native)

target_link_libraries(customop ${ONNXRUNTIME_LIBRARY})
