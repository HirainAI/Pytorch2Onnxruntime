cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# modify to your own nvcc path, or delete it if ok
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# set(SAMPLING_LIBRARY_PATH "/root/workspace/custom_test/pytorch/libsampling_gpu.so")
project(fps LANGUAGES CXX CUDA)
##find_package(Python REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
#find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

#set(CUDNN_INCLUDE_DIR=/opt/anaconda3/pkgs/cudnn-7.6.5-cuda9.2_0/include)
#set(CUDNN_LIBRARY=/opt/anaconda3/pkgs/cudnn-7.6.5-cuda9.2_0/lib)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# find_library(LIBRARY torch PATHS  "/root/anaconda3/lib/")

# modify to your own python path, or delete it if ok
include_directories(/usr/include/python3.6)
include_directories(../include)

set(SRCS sampling.cpp sampling_gpu.cu)
cuda_add_library(fps SHARED ${SRCS})
# LINK_DIRECTORIES(/root/workspace/custom_test/pytorch)
target_link_libraries(fps 
    # ${LIBRARY}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)