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
#include <pybind11/pybind11.h>

#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/api/python/functional/unpack_call.h"
#include "oneflow/api/python/framework/throw.h"

#include "oneflow/core/profiler/profiler.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

template<typename SchemaT>
static PyObject* PyFunction(PyObject* self, PyObject* args) {
  OF_PROFILER_RANGE_PUSH("PyFunction");

  // TODO(): Support multiple function signatures.
  size_t nargs = PyTuple_Size(args);
  CHECK_LE_OR_THROW(nargs, SchemaT::max_positionals)
      << "The maximum count of positional arguments is " << SchemaT::max_positionals;

  // TODO(): Check argument types.
  std::vector<PythonArg> _args(SchemaT::max_args);
  for (int i = 0; i < nargs; ++i) { _args[i] = PythonArg(PyTuple_GetItem(args, i)); }
  for (int i = nargs; i < SchemaT::max_args; ++i) {
    const auto& arg = SchemaT::argument_def.at(i);
    CHECK_OR_THROW(arg.has_default_value)
        << "Argument " << arg.name << " is required, and the function def is \""
        << SchemaT::signature << "\".";
    _args[i] = PythonArg(arg.default_value);
  }
  using FType = typename SchemaT::FType;
  using R = typename SchemaT::R;

  OF_PROFILER_RANGE_PUSH("unpack_call");
  auto result = py::cast(detail::unpack_call<FType, R>::apply(*SchemaT::func, _args));
  OF_PROFILER_RANGE_POP();

  OF_PROFILER_RANGE_POP();
  result.inc_ref();
  return result.ptr();
}

template<typename SchemaT>
static PyObject* PyFunctionWithKeywords(PyObject* self, PyObject* args, PyObject* kwargs) {
  OF_PROFILER_RANGE_PUSH("PyFunction");

  // TODO(): Support multiple function signatures.
  size_t nargs = PyTuple_Size(args);
  CHECK_LE_OR_THROW(nargs, SchemaT::max_positionals)
      << "The maximum count of positional arguments is " << SchemaT::max_positionals;
  size_t nkwargs = PyDict_Size(kwargs);
  CHECK_LE_OR_THROW(nkwargs, SchemaT::max_keywords)
      << "The maximum count of keyword arguments is " << SchemaT::max_keywords;

  // TODO(): Check argument types.
  std::vector<PythonArg> _args(SchemaT::max_args);
  for (int i = 0; i < nargs; ++i) { _args[i] = PythonArg(PyTuple_GetItem(args, i)); }
  for (int i = nargs; i < SchemaT::max_args; ++i) {
    const auto& arg = SchemaT::argument_def.at(i);
    PyObject* obj = PyDict_GetItemString(kwargs, arg.name.c_str());
    if (obj) {
      _args[i] = PythonArg(obj);
    } else {
      CHECK_OR_THROW(arg.has_default_value)
          << "Argument " << arg.name << " is required, and the function def is \""
          << SchemaT::signature << "\".";
      _args[i] = PythonArg(arg.default_value);
    }
  }
  using FType = typename SchemaT::FType;
  using R = typename SchemaT::R;

  OF_PROFILER_RANGE_PUSH("unpack_call");
  auto result = py::cast(detail::unpack_call<FType, R>::apply(*SchemaT::func, _args));
  OF_PROFILER_RANGE_POP();

  OF_PROFILER_RANGE_POP();
  result.inc_ref();
  return result.ptr();
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
