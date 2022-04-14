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

import os
import numpy as np
import time
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


class TestPixelShuffleError(flow.unittest.TestCase):
    def test_pixel_shuffle_4D_input_error(test_case):
        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.Exception
        ) as ctx:
            x = flow.ones((1, 8, 4, 4, 1), dtype=flow.float32)
            out = flow._C.pixel_shuffle(x, 2, 2)

        test_case.assertTrue(
            "Check failed: x->ndim() == 4 Only Accept 4D Tensor" in str(ctx.exception)
        )

    def test_pixel_shuffle_channel_divisble_error(test_case):
        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.Exception
        ) as ctx:
            x = flow.ones((1, 8, 4, 4), dtype=flow.float32)
            out = flow._C.pixel_shuffle(x, 2, 3)

        test_case.assertTrue(
            "Check failed: channel % (h_upscale_factor * w_upscale_factor) == 0 The channels of input tensor must be divisible by (upscale_factor * upscale_factor) or (h_upscale_factor * w_upscale_factor)"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
