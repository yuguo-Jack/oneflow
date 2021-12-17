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
#ifndef ONEFLOW_CORE_VM_EVENT_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_VM_EVENT_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/vm/event.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {
namespace vm {

class EventPhyInstrOperand : public PhyInstrOperand {
 public:
  EventPhyInstrOperand(const std::shared_ptr<std::shared_ptr<vm::Event>>& event,
                       Symbol<Device> device,
                       const intrusive::shared_ptr<LocalDepObject>& resource_dep)
      : event_(event), device_(device), resource_dep_(resource_dep), output_dependences_() {
    CHECK_NOTNULL(event);
    output_dependences_.push_back(device->mut_schedule_local_dep_object());
    output_dependences_.push_back(resource_dep_.get());
  }
  ~EventPhyInstrOperand() override = default;

  const std::shared_ptr<vm::Event>& event() const { return *event_; }

  std::shared_ptr<vm::Event>* mut_event() { return event_.get(); }

  const DependenceVector& input_dependences() const override {
    static thread_local DependenceVector empty{};
    return empty;
  }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

 private:
  std::shared_ptr<std::shared_ptr<vm::Event>> event_;
  Symbol<Device> device_;
  intrusive::shared_ptr<LocalDepObject> resource_dep_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_EVENT_PHY_INSTR_OPERAND_H_
