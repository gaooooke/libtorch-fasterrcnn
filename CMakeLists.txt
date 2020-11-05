# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# See https://pytorch.org/tutorials/advanced/cpp_frontend.html
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchscript_traced_mask_rcnn)

# set(Torch_DIR /home/ps/libtorch)
set(Torch_DIR /home/ps/libtorch/share/cmake/Torch) #指定libTorch位置(应该是有更好的办法安装)

include_directories(${Torch_INCLUDE_DIRS} /home/ps/libtorch/include /home/ps/libtorch/include/torch/csrc/api/include)

find_package(Torch REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message("Not found Torch")
endif()

add_executable(torchscript_traced_mask_rcnn torchscript_traced_mask_rcnn.cpp)
target_link_libraries(
  torchscript_traced_mask_rcnn
  "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET torchscript_traced_mask_rcnn PROPERTY CXX_STANDARD 14)
