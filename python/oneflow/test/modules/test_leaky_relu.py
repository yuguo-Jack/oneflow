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

import numpy as np

from oneflow.test_utils.automated_test_util import *
from scipy import special
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestLeakyReLUModule(flow.unittest.TestCase):
    @autotest()
    def test_leakyrelu_module_with_random_data(test_case):
        m = torch.nn.LeakyReLU(negative_slope=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_leakyrelu_module_with_half_random_data(test_case):
        m = torch.nn.LeakyReLU(negative_slope=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        x = x.to(torch.float16)
        y = m(x)
        return y

    @autotest()
    def test_leakyrelu_module_with_0dim_data(test_case):
        m = torch.nn.LeakyReLU(negative_slope=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()
