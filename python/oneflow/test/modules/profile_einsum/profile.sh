WARMUP_ITERS=2
RUN_ITERS=10
MIN_SIZE_FOR_EACH_DIM=100
MAX_SIZE_FOR_EACH_DIM=100
# TEST_CASE="alphaflod_usecase7"

# declare -a arr=("matrix_transpose" 
#                 "eltwise_mul" 
#                 "get_diagonal" 
#                 "batch_permute" 
#                 "reduce_sum" 
#                 "matrix_column_sum" 
#                 "matrix_vector_multiply"
#                 "matmul"
#                 "vector_inner_product"
#                 "eltwise_mul_then_reduce_sum"
#                 "vector_outer_product"
#                 "batch_matmul"
#                 "bilinear_transformation"
#                 "attention"
#                 "batch_matmul2"
#                 "batch_matmul3"
#                 "batch_matmul4"
#                 "batch_matrix_vector_multiply"
#                 "alphaflod_usecase1"
#                 "alphaflod_usecase2"
#                 "alphaflod_usecase3"
#                 "alphaflod_usecase4"
#                 "alphaflod_usecase5"
#                 "alphaflod_usecase6"
#                 "alphaflod_usecase7"
#                 "alphaflod_usecase8"
#                 "alphaflod_usecase9"
#                 "alphaflod_usecase10"
#                 "alphaflod_usecase11"
#                 )

declare -a arr=("get_diagonal"
                )

for test_case in "${arr[@]}"
do
    /home/liangdepeng/nsight-systems-2021.5.1/bin/nsys profile --stats=true -o ${test_case}_oneflow python3 einsum_oneflow.py --einsum_testcase ${test_case} --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
    # /home/liangdepeng/nsight-systems-2021.5.1/bin/nsys profile --stats=true -o ${test_case}_torch python3 einsum_pytorch.py --einsum_testcase ${test_case} --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
done
