#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSYSTEMOPINTERFACES_TD_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSYSTEMOPINTERFACES_TD_

include "mlir/IR/OpBase.td"

def PlaceholderSystemOpCompatibleInterface : OpInterface<"PlaceholderSystemOpCompatible"> {
  let description = [{
    Interface to handle confs like InputOpConf, OutputOpConf
  }];

  let methods = [
    StaticInterfaceMethod<"", "const std::string& ", "GetJobNameFromOpConf", (ins "const ::oneflow::OperatorConf&":$op_conf),
        [{ return ConcreteOp::GetOneOfConf(op_conf).job_name(); }]
    >,
    StaticInterfaceMethod<"", "const std::string& ", "GetJobNameFromOpConf", (ins "const ::oneflow::OperatorConf&":$op_conf),
        [{ return ConcreteOp::GetOneOfConf(op_conf).job_name(); }]
    >
  ];
}

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWSYSTEMOPINTERFACES_TD_
