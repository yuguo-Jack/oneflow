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
from collections import OrderedDict
import oneflow
import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *
import time


def _test_nccl_slice_boxing_copy(test_case, shape):
    src_nd_sbp_str = ["P", "S(1)"]
    src_nd_sbp = [flow.sbp.partial_sum(), flow.sbp.split(1)]
    dst_nd_sbp = [flow.sbp.broadcast(), flow.sbp.split(2)]
    dst_nd_sbp_str = ["B", "S(2)"]
    placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            flow.boxing.nccl.enable_all_to_all(True)
            super().__init__()

        def build(self, x):
            y = x
            y = y.to_global(sbp=dst_nd_sbp, placement=placement)
            # y = y.to_global(sbp=src_nd_sbp, placement=placement)
            return y

    class TestGraph2(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            y = flow._C.nccl_slice_boxing_copy(x, src_nd_sbp_str, dst_nd_sbp_str)
            y = flow._C.nccl_slice_boxing_copy(y, dst_nd_sbp_str, src_nd_sbp_str)
            return y

    x = flow.tensor(
        np.arange(64 * 1024 * 1024).reshape(64, 1024, 1024),
        sbp=src_nd_sbp,
        placement=placement,
    )
    print("x", x.sbp, x.shape)
    # y = flow._C.nccl_slice_boxing_copy(x, src_nd_sbp, dst_nd_sbp)
    # graph = TestGraph2()
    graph = TestGraph()
    start_time = time.time()
    for i in range(1):
        y = graph(x)
    test_case.assertTrue(np.array_equal(y.numpy(), x.numpy()))


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestNcclSliceBoxingCopy(flow.unittest.TestCase):
    def test_nccl_slice_boxing_copy(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(12, 24, 12)]
        for arg in GenArgList(arg_dict):
            _test_nccl_slice_boxing_copy(test_case, arg)


if __name__ == "__main__":
    unittest.main()
