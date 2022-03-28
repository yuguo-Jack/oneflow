
import unittest
from collections import OrderedDict
import oneflow
import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *


def _test_nccl_slice_boxing_copy(test_case, shape):
    src_nd_sbp = ["S(0)", "S(1)", "S(2)"]
    dst_nd_sbp = ["S(1)", "S(2)", "S(0)"]

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            y = flow._C.nccl_slice_boxing_copy(x, src_nd_sbp, dst_nd_sbp)
            return y

    x = flow.tensor(
        np.arange(12*12*12).reshape(12, 12, 12),
        sbp=[flow.sbp.split(0), flow.sbp.split(1), flow.sbp.split(2)],
        placement=flow.placement("cuda", ranks=[[[0, 1], [2, 3]]]),
    )
    print("x", x.sbp,x.shape)
    y = flow._C.nccl_slice_boxing_copy(x, src_nd_sbp, dst_nd_sbp)
    #graph = TestGraph()
    #y=graph(x)
    print(y.numpy())
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
