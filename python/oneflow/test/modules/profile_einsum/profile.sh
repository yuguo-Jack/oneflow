
WARMUP_ITERS=2
RUN_ITERS=10
MIN_SIZE_FOR_EACH_DIM=100
MAX_SIZE_FOR_EACH_DIM=100
TEST_CASE="alphaflod_usecase7"

/home/liangdepeng/nsight-systems-2021.5.1/bin/nsys profile --stats=true -o einsum_oneflow.qdrep python3 einsum_oneflow.py --einsum_testcase $TEST_CASE --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM

/home/liangdepeng/nsight-systems-2021.5.1/bin/nsys profile --stats=true -o einsum_pytorch.qdrep python3 einsum_pytorch.py --einsum_testcase $TEST_CASE --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM

