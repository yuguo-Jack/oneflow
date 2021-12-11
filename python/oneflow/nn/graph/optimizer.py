"""
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
"""
from oneflow.nn.optimizer.optimizer import Optimizer
from oneflow.nn.optimizer.sparse_optimizer import SparseOptimizer
from oneflow.nn.optimizer.lr_scheduler import LrScheduler


class OptDict(object):
    def __init__(self, opt_dict):
        if not isinstance(opt_dict, dict):
            raise ValueError("opt_dict is not a dict")

        if "optim" in opt_dict:
            if isinstance(opt_dict["optim"], Optimizer):
                self._optimizer = opt_dict["optim"]
                self._is_sparse = False
            elif isinstance(opt_dict["optim"], SparseOptimizer):
                self._optimizer = opt_dict["optim"]._nested_optim
                self._is_sparse = True
            else:
                raise ValueError(
                    'opt_dict["optim"] is not an instance of Optimizer or SparseOptimizer'
                )
        else:
            raise ValueError("opt_dict has not key 'optim'")

        self._lr_scheduler = None
        if "lr_sch" in opt_dict:
            if not isinstance(opt_dict["lr_sch"], LrScheduler):
                raise ValueError('opt_dict["lr_sch"] is not an instance of LrScheduler')

            if opt_dict["lr_sch"]._optimizer is not self._optimizer:
                raise ValueError("lr_scheduler's optimizer is not same with optimizer")

            self._lr_scheduler = opt_dict["lr_sch"]

    def generate_optimizer_and_variable_configs(self, job_conf, vars_conf):
        train_conf = job_conf.mutable_train_conf()

        if self._optimizer is not None:
            opt_confs = self._optimizer._generate_conf_for_graph(train_conf, vars_conf)
            self._optimizer._check_variables_optimizer_bound(vars_conf)

            if self._is_sparse:
                self._optimizer._generate_indexed_slices_optimizer_conf(
                    job_conf, vars_conf
                )

        if self._lr_scheduler is not None:
            self._lr_scheduler._generate_conf_for_graph(opt_confs)


class VariableConfig(object):
    def __init__(self, name: str):
        assert name != ""
        self._name = name
        self._l2 = 0.0
        self._bound_opt = None

    @property
    def name(self):
        return self._name

    @property
    def l2(self):
        return self._l2

    @property
    def bound_optimizer(self):
        return self._bound_opt

    @bound_optimizer.setter
    def bound_optimizer(self, opt):
        self._bound_opt = opt

    @l2.setter
    def l2(self, l2: float = 0.0):
        self._l2 = l2

    def __repr__(self):
        return "(variable name: " + self._name + "):(l2: " + str(self._l2) + ".)"
