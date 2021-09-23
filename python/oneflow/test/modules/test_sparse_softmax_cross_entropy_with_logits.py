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

import unittest
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import oneflow as flow
import oneflow.unittest
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def compare_distributed_with_tensorflow(
        data_type, label_type, batch_size, num_classes, 
    ):
        device_type = "cuda"
        data_type = type_name_to_flow_type[data_type]
        label_type = type_name_to_flow_type[label_type]
        np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
        np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)
        placement = flow.placement(device_type, {0: range(4)})

        of_logits = flow.tensor(np_logits, device=device_type, dtype=data_type, requires_grad=True)
        flow.comm.broadcast(of_logits, 0)
        of_logits = of_logits.to_consistent(placement=placement, sbp=[flow.sbp.broadcast])
        of_logits = of_logits.to_consistent(placement=placement, sbp=[flow.sbp.split(1)])
        of_labels = flow.tensor(np_labels, device=device_type, dtype=label_type)
        flow.comm.broadcast(of_labels, 0)
        of_labels = of_labels.to_consistent(placement=placement, sbp=[flow.sbp.broadcast])
        
        of_output = flow.nn.functional.distributed_sparse_softmax_cross_entropy_with_logits(
            labels=of_labels, logits=of_logits
        ).to(device_type)
        of_output = of_output.to_consistent(placement=placement, sbp=[flow.sbp.broadcast])
        of_output = of_output.to_local()

        of_output.sum().backward()
        of_logits_grad = of_logits.grad.to_consistent(placement=placement, sbp=[flow.sbp.broadcast])
        of_logits_grad = of_logits_grad.to_local()

class TestSparseSoftmaxCrossEntropyWithLogitsGrid(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_distributed_sparse_softmax_cross_entropy_with_logits(test_case):
        compare_distributed_with_tensorflow("float32", "int32", 64, 100)


if __name__ == "__main__":
    unittest.main()
