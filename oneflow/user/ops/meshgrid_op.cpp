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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("meshgrid")
    .InputWithMinimum("in", 1)
    .OutputWithMinimum("out",1)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const int64_t in_size = ctx->input_size("in");
      CHECK_EQ_OR_RETURN(ctx->output_size("out"), in_size);
      
      DimVector out_dim_vec = {};
      for (int64_t i = 0; i < in_size; ++i){
          const auto& cur_in = ctx->InputTensorDesc("in", i);
          CHECK_LE_OR_RETURN(cur_in.shape().NumAxes(),1);
          if(cur_in.shape().NumAxes()==0){
              out_dim_vec.push_back(1);
          }else{
              out_dim_vec.push_back(cur_in.shape().At(0));
          }
      }
      for (int64_t i = 0; i < in_size; ++i){
          const auto& cur_out = ctx->OutputTensorDesc("out", i);
          *cur_out->mut_shape() = Shape(out_dim_vec);
      }
      return Maybe<void>::Ok();
    })
   .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const int64_t in_size = ctx->input_size("in");
      CHECK_EQ_OR_RETURN(ctx->output_size("out"), in_size);
      for (int64_t i = 0; i < in_size; ++i) {
        *ctx->OutputDType("out", i) = ctx->InputDType("in", i);
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

REGISTER_USER_OP("meshgrid_grad")
    .Input("dy")
    .Output("dx")
    .Attr<int32_t>("index")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("dx", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

REGISTER_USER_OP_GRAD("meshgrid").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) -> Maybe<void> {
  int32_t in_size = op.input_size("in");
  for (int i = 0; i < in_size; ++i) {
    if (op.NeedGenGradTensor4OpInput("in", i)) {
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
      user_op::UserOpConfWrapper meshgrid_grad_op =
            builder.Op("meshgrid_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("out", i))
                .Output("dx")
                .Build();
      op.BindGradTensorWithOpInput(meshgrid_grad_op.output("dx", 0), "in", i);
      AddOp(meshgrid_grad_op);
    }
  }
  return Maybe<void>::Ok();
});

// REGISTER_USER_OP_GRAD("meshgrid").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
//                                                          user_op::AddOpFn AddOp) -> Maybe<void> {
//   int32_t in_size = op.input_size("in");
//   for (int i = 0; i < in_size; ++i) {
//     if (op.NeedGenGradTensor4OpInput("in", i)) {
//       user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
//       user_op::UserOpConfWrapper meshgrid_grad_op =
//             builder.Op("meshgrid_grad")
//                 .Input("out", op.output("out", i))
//                 .Input("dy", op.GetGradTensorWithOpOutput("out", i))
//                 .Input("in",op.output("in",i))
//                 .Output("dx")
//                 .Build();
//       op.BindGradTensorWithOpInput(meshgrid_grad_op.output("dx", 0), "in", i);
//       AddOp(meshgrid_grad_op);
//     }
//   }
//   return Maybe<void>::Ok();
// });

}  // namespace oneflow