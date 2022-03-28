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
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/impl/binary_functor.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class LeakyReluFunctor {
 public:
  LeakyReluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("leaky_relu").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("alpha", alpha));
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LeakyReluGradFunctor {
 public:
  LeakyReluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("leaky_relu_grad").Input("x").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& dy, const float& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("alpha", alpha));
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::LeakyReluFunctor>("LeakyRelu");
  m.add_functor<impl::LeakyReluGradFunctor>("LeakyReluGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
