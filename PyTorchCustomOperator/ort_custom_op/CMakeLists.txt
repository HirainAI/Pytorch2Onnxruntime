cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)

set(ONNXRUNTIME_ROOT_PATH /root/workspace/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS  /root/workspace/onnxruntime/include/onnxruntime)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)

set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.h) 
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})
#Include path to header files for Custom Op
include_directories(/root/workspace/pytorch/third_party/eigen)
include_directories(ONNXRUNTIME_INCLUDE_DIRS)

#Include path to header files for Custom Op Test
#include_directories(/tmp/onnxInstall/runtimes/linux-x64/native)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS  ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
