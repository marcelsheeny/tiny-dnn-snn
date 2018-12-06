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

// void changeDataLayout(const vec_t &input,
//                       vec_t &output,
//                       const int &N,
//                       const int &C,
//                       const int &W,
//                       const int &H) {
//   for (int n = 0; n < N; n++) {
//     for (int c = 0; c < C; c++) {
//       for (int w = 0; w < W; w++) {
//         for (int h = 0; h < H; h++) {
//           input[n * N + c * C + w * W + h] = output[n * N + c * C + w * W +
//           h];
//         }
//       }
//     }
//   }

void add_bias(const vec_t &bias,
              const int &width,
              const int &height,
              tensor_t &tensor) {
  // for each data of the bacth
  for (int b = 0; b < tensor.size(); b++) {
    // for each bias point
    for (int i = 0; i < bias.size(); i++) {
      // for each data point
      for (int j = 0; j < height; j++) {
        for (int k = 0; k < width; k++) {
          int tensor_index = i * bias.size() + j * height + k;
          tensor[b][tensor_index] += bias[i];
        }
      }
    }
  }
}

void vector2d_to_vector1d(const tensor_t &vec2d, vec_t &vec1d) {
  for (int i = 0; i < vec2d.size(); ++i) {
    const vec_t &v = vec2d[i];
    vec1d.insert(vec1d.end(), v.begin(), v.end());
  }
}

void vector1d_to_vector2d(const vec_t &vec1d, tensor_t &vec2d) {
  const int batch = vec1d.size() / vec2d.size();
  for (int i = 0; i < vec2d.size(); ++i) {
    int init_index              = i * batch;
    vec_t::const_iterator first = vec1d.begin() + init_index;
    vec_t::const_iterator last  = vec1d.begin() + init_index + batch;
    vec_t subvec(first, last);
    vec2d[i] = subvec;
  }
}

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
  conv_params.in_rows       = params.in_padded.height_;
  conv_params.in_cols       = params.in_padded.width_;
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
  vec_t input;
  for (auto &&v : in_data) {
    input.insert(input.end(), v.begin(), v.end());
  }

  // copy host to device
  device.memcpyHostToDevice(input_gpu_buffer, input.data(), input_nbytes);
  device.memcpyHostToDevice(filter_gpu_buffer, W.data(), filter_nbytes);

  // Now that all of our buffers are populated, and parameters configured,
  // we can execute the convolution itself. This happens asynchronously, so
  // we follow the launch of the convolution kernel with a blocking wait.
  auto direct_algo_selector = sycldnn::conv2d::DirectSelector{};

  auto status =
    sycldnn::conv2d::launch<value_type, sycldnn::conv2d::conv_type::Forward>(
      input_gpu_buffer, filter_gpu_buffer, output_gpu_buffer, conv_params,
      direct_algo_selector, backend);

  // resize output vector
  vec_t output;
  output.resize(conv_sizes.output_size);

  // Wait for completion, then copy results to system memory.
  status.event.wait();
  device.memcpyDeviceToHost(output.data(), output_gpu_buffer, output_nbytes);

  // The convolution results are now available in host-accessible system
  // memory.
  // copy to output tensor
  vector1d_to_vector2d(output, out_data);

  if (params.has_bias)
    add_bias(bias, params.out.width_, params.out.height_, out_data);

  // We can now deallocate the Eigen GPU buffers.
  device.deallocate(input_gpu_buffer);
  device.deallocate(output_gpu_buffer);
  device.deallocate(filter_gpu_buffer);
}

}  // namespace kernels
}  // namespace tiny_dnn
