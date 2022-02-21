
WARMUP_ITERS=5
RUN_ITERS=100
MIN_SIZE_FOR_EACH_DIM=100
MAX_SIZE_FOR_EACH_DIM=100

python3 einsum_speed_test.py --einsum_testcase "matrix_transpose" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "eltwise_mul" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "get_diagonal" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "batch_permute" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "reduce_sum" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "matrix_column_sum" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "matrix_vector_multiply" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "matmul" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "vector_inner_product" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "eltwise_mul_then_reduce_sum" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "vector_outer_product" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "batch_matmul" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "tensor_contraction" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim 20 --max_size_for_each_dim 20
python3 einsum_speed_test.py --einsum_testcase "bilinear_transformation" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "attention" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "batch_matmul2" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "batch_matmul3" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "batch_matmul4" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "batch_matrix_vector_multiply" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase1" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase2" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase3" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase4" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase5" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase6" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase7" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase8" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase9" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase10" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM
python3 einsum_speed_test.py --einsum_testcase "alphaflod_usecase11" --warm_up_iters $WARMUP_ITERS --run_iters $RUN_ITERS --min_size_for_each_dim $MIN_SIZE_FOR_EACH_DIM --max_size_for_each_dim $MAX_SIZE_FOR_EACH_DIM


