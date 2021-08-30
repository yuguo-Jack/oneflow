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

#ifndef ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_H_
#define ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_H_

#include <memory>

#include "oneflow/core/common/function_traits.h"
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/functional/value_types.h"
#include "oneflow/core/functional/function_signature.h"

namespace oneflow {
namespace one {
namespace functional {

template<typename T>
using remove_cvref_t = oneflow::detail::remove_cvref_t<T>;

struct FunctionBody {
  virtual operator const void*() const = 0;
  virtual std::type_index signature() const = 0;
  virtual ~FunctionBody() = default;

  template<typename R, typename... Args>
  R call(const remove_cvref_t<Args>&... args) const {
    if (signature() != typeid(R(const remove_cvref_t<Args>&...))) {
      LOG(FATAL) << "The function was called with wrong arguments.";
    }
    using FuncType = std::function<R(const remove_cvref_t<Args>&...)>;
    auto* func = reinterpret_cast<const FuncType*>(this->operator const void*());
    return (*func)(std::forward<const remove_cvref_t<Args>&>(args)...);
  }
};

template<typename T>
class FunctionBodyImpl;

template<typename R, typename... Args>
class FunctionBodyImpl<R(Args...)> : public FunctionBody {
 public:
  template<typename Func,
           typename std::enable_if<
               std::is_same<typename function_traits<Func>::func_type, R(Args...)>::value,
               int>::type = 0>
  FunctionBodyImpl(const Func& func) : func_([func](const remove_cvref_t<Args>&... args) {
      return func(std::forward<const remove_cvref_t<Args>&>(args)...);
    }), sig_(typeid(R(const remove_cvref_t<Args>&...))) {}

  operator const void*() const override { return &func_; }
  std::type_index signature() const override { return sig_; }

 private:
  std::function<R(const remove_cvref_t<Args>&...)> func_;
  std::type_index sig_;
};

class Functor {
 public:
  Functor(const std::shared_ptr<FunctionBody>& body)
      : body_(body) {}

  template<typename R, typename... Args>
  R call(const remove_cvref_t<Args>&... args) const {
    return body_->call<R, Args...>(std::forward<const remove_cvref_t<Args>&>(args)...);
  }

 private:
  std::shared_ptr<FunctionBody> body_;
};

class PackedFunctor {
 public:
  PackedFunctor(const std::string& func_name, const Functor& functor)
      : func_name_(func_name), functor_(functor) {}

  virtual ~PackedFunctor() = default;

  template<typename Func>
  static PackedFunctor Make(const std::string& func_name, const Func& func);

  template<typename R, typename... Args>
  R call(Args... args) const {
    return functor_.call<R, Args...>(std::forward<Args>(args)...);
  }

 private:
  std::string func_name_;
  Functor functor_;
};

template<typename Func>
PackedFunctor PackedFunctor::Make(const std::string& func_name, const Func& func) {
  using func_type = typename function_traits<Func>::func_type;
  Functor functor(std::make_shared<FunctionBodyImpl<func_type>>(func));
  return PackedFunctor(func_name, std::move(functor));
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_H_
