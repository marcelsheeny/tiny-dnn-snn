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

cmake_minimum_required(VERSION 3.2.2)


if(DOWNLOAD_SYCLDNN)
  message(STATUS "Configuring SYCL-DNN library")
  # Select a commit from the Eigen-SYCL-OpenCL branch. This should be manually
  # bumped as appropriate.
  set(SYCL_GIT_TAG "a54f3e9" CACHE STRING
    "git tag, branch or commit to use for the SYCL-DNN library"
  )

  configure_file(${CMAKE_SOURCE_DIR}/cmake/Modules/DownloadSYCLDNN.cmake.in
    ${CMAKE_BINARY_DIR}/sycldnn-download/CMakeLists.txt)

  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/sycldnn-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for SYCL-DNN failed: ${result}")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/sycldnn-download
  )
  if(result)
    message(FATAL_ERROR "Download step for SYCL-DNN failed: ${result}")
  endif()
endif()

include_directories(${CMAKE_BINARY_DIR}/SYCLDNN-src/include)

