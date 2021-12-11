module  {
  "oneflow.job"() ( {
  ^bb0(%arg0: tensor<2x2xf32>):  // no predecessors
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_InferGraph_0-input_0", output_lbns = ["_InferGraph_0-input_0/out"], scope_symbol_id = 4611686018427449343 : i64, shape = dense<2> : vector<2xi64>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %output_0 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], nd_sbp = ["B"], op_name = "model.weight", output_lbns = ["model.weight/out"], scope_symbol_id = 4611686018427465727 : i64, shape = dense<2> : vector<2xi64>} : () -> tensor<2x2xf32>
    %0 = "oneflow.add_n2"(%output, %output_0) {device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], op_name = "model-add_n_0", op_type_name = "add_n", output_lbns = ["model-add_n_0/out_0"], scope_symbol_id = 4611686018427469823 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %output_1 = "oneflow.output"(%0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_InferGraph_0-output_0", output_lbns = ["_InferGraph_0-output_0/out"], scope_symbol_id = 4611686018427473919 : i64, shape = dense<2> : vector<2xi64>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    oneflow.return %output_1 : tensor<2x2xf32>
  }) {sym_name = "InferGraph_0", type = (tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}
