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
#include <cstdint>
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct MeshgridCaptureState : public AutoGradCaptureState {
   std::vector<bool> requires_grad;
};

class Meshgrid : public OpExprGradFunction<MeshgridCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MeshgridCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const MeshgridCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override; 
  private:
    AttrMap base_attrs_;
};

Maybe<void> Meshgrid::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Meshgrid::Capture(MeshgridCaptureState* ctx, const TensorTuple& inputs,
                          const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), outputs.size());
  for(int32_t i=0; i<inputs.size(); i++){
      if(inputs.at(i)->requires_grad()){
          ctx->requires_grad.push_back(true);
      }else{
          ctx->requires_grad.push_back(false);
      }
  }
  return Maybe<void>::Ok();
}

Maybe<void> Meshgrid::Apply(const MeshgridCaptureState* ctx, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const {
  for(int32_t i=0; i < ctx->requires_grad.size();i++){
      if(ctx->requires_grad[i]){
          in_grads->at(i) = JUST(functional::MeshgridGrad(out_grads.at(i),i));
      }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("meshgird", Meshgrid);

}  // namespace one
}  // namespace oneflow