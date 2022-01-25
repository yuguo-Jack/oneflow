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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=False)
def _test_add_with_alpha(test_case, placement, sbp):
    try:
        data = flow.rand(2, dtype=flow.float32)
        consistent_data = data.to_consistent(placement=placement, sbp=sbp)

        if flow.env.get_rank() == 0:
            print(data.mean())
            print(consistent_data.mean())

    except Exception as e:
        print(
            "Failed to apply add operation on x and y with (placement: %s, sbp: %s)"
            % (placement.value(), sbp.value()),
        )
        print(str(e))


class TestAddModule(flow.unittest.TestCase):
    @consistent
    def test_add_with_alpha(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_add_with_alpha(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()