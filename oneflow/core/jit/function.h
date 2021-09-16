#ifndef ONEFLOW_CORE_JIT_FUNCTION_H_
#define ONEFLOW_CORE_JIT_FUNCTION_H_

#include <memory>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/vm/instruction.msg.h"

namespace oneflow {
namespace jit {

class Function final : public std::enable_shared_from_this<Function> {
 public:
  Function(const Function&) = delete;
  Function(Function&&) = delete;
  Function() : input_placeholders_(NullOpt), output_placeholders_(NullOpt), instruction_msg_list_(NullOpt) {}
  ~Function() = default;

  static Maybe<Function> FindOrCreate(const TensorTuple& inputs);

  Maybe<TensorTuple> operator()(const TensorTuple& inputs) const;

  Maybe<void> SetInputPlaceholders(const std::shared_ptr<TensorTuple>& input_placeholders) {
    CHECK_OR_RETURN(!input_placeholders_.has_value());
    input_placeholders_ = input_placeholders;
    return Maybe<void>::Ok();
  }

  Maybe<void> SetOutputPlaceholders(const std::shared_ptr<TensorTuple>& output_placeholders) {
    CHECK_OR_RETURN(!output_placeholders_.has_value());
    output_placeholders_ = output_placeholders;
    return Maybe<void>::Ok();
  }
  
  Maybe<void> MoveInstructionMsgListFrom(vm::InstructionMsgList* instruction_msg_list) {
    CHECK_OR_RETURN(!instruction_msg_list_.has_value());
    instruction_msg_list_ = std::make_shared<vm::InstructionMsgList>();
    instruction_msg_list->MoveTo(JUST(instruction_msg_list_).get());
    return Maybe<void>::Ok();
  }

 private:
  Maybe<TensorTuple> input_placeholders() const { return JUST(input_placeholders_); }
  Maybe<TensorTuple> output_placeholders() const { return JUST(output_placeholders_); }

  Optional<TensorTuple> input_placeholders_;
  Optional<TensorTuple> output_placeholders_;
  Optional<vm::InstructionMsgList> instruction_msg_list_;
};

}
}

#endif  // ONEFLOW_CORE_JIT_FUNCTION_H_
