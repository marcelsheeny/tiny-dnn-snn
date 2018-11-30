
# Copyright 2018 Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# snn_executable(
#   WITH_SYCL
#   TARGET
#     conv2d
#   KERNEL_SOURCES
#     test_sycldnn.cpp
#   PRIVATE_LIBRARIES
#     sycl_dnn_static
#     Eigen::Eigen
# )

set(SOURCE_NAME "test_sycldnn")

add_executable(
  ${SOURCE_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)

target_link_libraries(${SOURCE_NAME} SYCLDNN::sycl_dnn)

add_sycl_to_target(
  TARGET ${SOURCE_NAME}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)

install(
  TARGETS ${SOURCE_NAME}
  RUNTIME DESTINATION bin
)