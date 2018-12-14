/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

#include <cassert>
#include <memory>
#include <numeric>
#include <unsupported/Eigen/CXX11/Tensor>

#include <sycldnn/backend/eigen_backend.h>
#include <sycldnn/matmul/launch.h>

namespace tiny_dnn {
namespace kernels {

void add_bias(const vec_t &bias, const int &out_size, tensor_t &tensor) {
  // for each data of the batch
  for (int b = 0; b < tensor.size(); b++) {
    // for each bias point
    for (int i = 0; i < bias.size(); i++) {
      // for each data point
      for (int j = 0; j < out_size; j++) {
        int tensor_index = i * bias.size() + out_size;
        tensor[b][tensor_index] += bias[i];
      }
    }
  }
}

inline void fully_connected_op_sycldnn(const tensor_t &in_data,
                                       const vec_t &W,
                                       const vec_t &bias,
                                       tensor_t &out_data,
                                       const core::fully_params &params,
                                       const bool layer_parallelize) {
  // select device
  auto device_selector = cl::sycl::default_selector{};

  // create queue
  auto queue = std::unique_ptr<Eigen::QueueInterface>(
    new Eigen::QueueInterface{device_selector});
  auto device = Eigen::SyclDevice{queue.get()};

  // select the sycl dnn backend
  auto backend = sycldnn::backend::EigenBackend{device};

  auto batch_size = in_data.size();

  // allocate buffers
  using value_type   = float;
  auto input_nbytes  = params.in_size_ * batch_size * sizeof(value_type);
  auto output_nbytes = params.out_size_ * batch_size * sizeof(value_type);
  auto weights_nbytes =
    params.in_size_ * params.out_size_ * batch_size * sizeof(value_type);

  auto *input_gpu_buffer =
    static_cast<value_type *>(device.allocate(input_nbytes));
  auto *output_gpu_buffer =
    static_cast<value_type *>(device.allocate(output_nbytes));
  auto *weights_gpu_buffer =
    static_cast<value_type *>(device.allocate(weights_nbytes));

  // populate input buffer
  vec_t input;
  for (auto &&v : in_data) {
    input.insert(input.end(), v.begin(), v.end());
  }

  // copy host to device
  device.memcpyHostToDevice(input_gpu_buffer, input.data(), input_nbytes);
  device.memcpyHostToDevice(weights_gpu_buffer, W.data(), weights_nbytes);

  // Now that all of our buffers are populated, and parameters configured,
  // we can execute the convolution itself. This happens asynchronously, so
  // we follow the launch of the convolution kernel with a blocking wait.

  auto status = sycldnn::matmul::launch<value_type, false, false>(
    weights_gpu_buffer, input_gpu_buffer, output_gpu_buffer, batch_size,
    params.out_size_, params.in_size_, 1, 1, backend);

  // resize output vector
  vec_t output;
  output.resize(params.out_size_);

  // Wait for completion, then copy results to system memory.
  status.event.wait();
  device.memcpyDeviceToHost(output.data(), output_gpu_buffer, output_nbytes);

  // The matmul results are now available in host-accessible system
  // memory.
  // copy to output tensor
  vector1d_to_vector2d(output, out_data);

  add_bias(bias, params.out_size_, out_data);

  // We can now deallocate the Eigen GPU buffers.
  device.deallocate(input_gpu_buffer);
  device.deallocate(output_gpu_buffer);
  device.deallocate(weights_gpu_buffer);
}

}  // namespace kernels
}  // namespace tiny_dnn
