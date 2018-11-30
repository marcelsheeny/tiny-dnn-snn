/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cassert>
#include <memory>
#include <numeric>
#include <unsupported/Eigen/CXX11/Tensor>

#include <sycldnn/backend/eigen_backend.h>
#include <sycldnn/conv2d/launch.h>
#include <sycldnn/conv2d/params.h>
#include <sycldnn/conv2d/selector/direct_selector.h>
#include <sycldnn/conv2d/selector/im2col_selector.h>
#include <sycldnn/conv2d/selector/winograd_selector.h>
#include <sycldnn/conv2d/sizes.h>

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_sycldnn(const tensor_t &in_data,
                              const vec_t &W,
                              const vec_t &bias,
                              tensor_t &out_data,
                              const core::conv_params &params,
                              const bool parallelize) {
  // select device
  auto device_selector = cl::sycl::default_selector{};

  // create queue
  auto queue = std::unique_ptr<Eigen::QueueInterface>(
    new Eigen::QueueInterface{device_selector});
  auto device = Eigen::SyclDevice{queue.get()};

  // select the sycl dnn backend
  auto backend = sycldnn::backend::EigenBackend{device};

  // init conv2d params
  sycldnn::conv2d::Conv2DParams conv_params{};
  conv_params.channels      = params.in.depth_;
  conv_params.features      = params.out.depth_;
  conv_params.batch         = in_data.size();
  conv_params.in_rows       = params.in.height_;
  conv_params.in_cols       = params.in.width_;
  conv_params.window_rows   = params.weight.height_;
  conv_params.window_cols   = params.weight.width_;
  conv_params.stride_rows   = params.h_stride;
  conv_params.stride_cols   = params.w_stride;
  conv_params.out_rows      = params.out.height_;
  conv_params.out_cols      = params.out.width_;
  conv_params.pad_rows      = 0;
  conv_params.pad_cols      = 0;
  conv_params.dilation_rows = params.h_dilation;
  conv_params.dilation_cols = params.w_dilation;

  // automatically compute sizes
  auto conv_sizes =
    sycldnn::conv2d::get_sizes<sycldnn::conv2d::conv_type::Forward>(
      conv_params);

  // allocate buffers
  using value_type   = float;
  auto input_nbytes  = conv_sizes.input_size * sizeof(value_type);
  auto output_nbytes = conv_sizes.output_size * sizeof(value_type);
  auto filter_nbytes = conv_sizes.filter_size * sizeof(value_type);

  auto *input_gpu_buffer =
    static_cast<value_type *>(device.allocate(input_nbytes));
  auto *output_gpu_buffer =
    static_cast<value_type *>(device.allocate(output_nbytes));
  auto *filter_gpu_buffer =
    static_cast<value_type *>(device.allocate(filter_nbytes));

  // populate input buffer
  std::vector<value_type> input;
  // input.resize(conv_sizes.input_size);
  for (auto &&v : in_data) {
    input.insert(input.end(), v.begin(), v.end());
  }

  // copy host to device
  device.memcpyHostToDevice(input_gpu_buffer, input.data(), input_nbytes);
  device.memcpyHostToDevice(filter_gpu_buffer, W.data(), filter_nbytes);

  // Now that all of our buffers are populated, and parameters configured, we
  // can execute the convolution itself. This happens asynchronously, so we
  // follow the launch of the convolution kernel with a blocking wait.
  auto direct_algo_selector = sycldnn::conv2d::DirectSelector{};
  auto status =
    sycldnn::conv2d::launch<value_type, sycldnn::conv2d::conv_type::Forward>(
      input_gpu_buffer, filter_gpu_buffer, output_gpu_buffer, conv_params,
      direct_algo_selector, backend);

  // populate output vector
  std::vector<value_type> output;
  output.resize(conv_sizes.output_size);
  for (auto &&v : out_data) {
    output.insert(output.end(), v.begin(), v.end());
  }

  // Wait for completion, then copy results to system memory.
  status.event.wait();
  device.memcpyDeviceToHost(output.data(), output_gpu_buffer, output_nbytes);

  // The convolution results are now available in host-accessible system
  // memory.

  // We can now deallocate the Eigen GPU buffers.
  device.deallocate(input_gpu_buffer);
  device.deallocate(output_gpu_buffer);
  device.deallocate(filter_gpu_buffer);
}

}  // namespace kernels
}  // namespace tiny_dnn
