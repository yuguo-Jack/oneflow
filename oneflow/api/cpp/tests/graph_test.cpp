/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <gtest/gtest.h>
#include <cstdint>
#include "oneflow/api/cpp/framework/graph.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/api/cpp/tests/api_test.h"

namespace oneflow_api {

TEST(Api, graph_test) {
  EnvScope scope;

  const std::string file_name = __FILE__;
  const std::string directory = file_name.substr(0, file_name.rfind('/'));

  Graph graph = Load(directory + "/graph_test_model");
  std::vector<Tensor> inputs;
  inputs.emplace_back(Shape{2, 2});
  inputs[0].zeros_();

  Tensor output = graph.Forward(inputs).at(0);
  Shape shape = output.shape();
  ASSERT_EQ(shape.At(0), 2);
  ASSERT_EQ(shape.At(1), 2);
  std::array<float, 4> buf{};
  output.copy_to(buf.data());
  ASSERT_EQ(buf[0], 1);
  ASSERT_EQ(buf[1], 1);
  ASSERT_EQ(buf[2], 1);
  ASSERT_EQ(buf[3], 1);
}

}  // namespace oneflow_api
