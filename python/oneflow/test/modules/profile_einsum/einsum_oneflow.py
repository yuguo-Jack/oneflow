import oneflow as flow

import time
import random
import argparse


def parse_option():
    parser = argparse.ArgumentParser("einsum speed test", add_help=False)
    parser.add_argument("--warm_up_iters", type=int, default=5, help="warm up iters")
    parser.add_argument("--run_iters", type=int, default=10, help="cal time iters")
    parser.add_argument(
        "--min_size_for_each_dim", type=int, default=100, help="random size range"
    )
    parser.add_argument(
        "--max_size_for_each_dim", type=int, default=100, help="random size range"
    )
    parser.add_argument("--einsum_testcase", type=str, default="alphaflod_usecase7", help="random size range")
    parser.add_argument("--ellipsis_dims", type=int, default=2, help="cal time iters")

    args, unparsed = parser.parse_known_args()

    return args


args = parse_option()


usecase_to_equations = {
    "matrix_transpose": "ij->ji",
    "eltwise_mul": "ij,ij->ij",
    "get_diagonal": "ii->i",
    "batch_permute": "...ij->...ji",
    "reduce_sum": "ij->",
    "matrix_column_sum": "ij->j",
    "matrix_vector_multiply": "ik,k->i",
    "matmul": "ik,kj->ij",
    "vector_inner_product": "i,i->",
    "eltwise_mul_then_reduce_sum": "ij,ij->",
    "vector_outer_product": "i,j->ij",
    "batch_matmul": "ijk,ikl->ijl",
    "tensor_contraction": "pqrs,tuqvr->pstuv",
    "bilinear_transformation": "ik,jkl,il->ij",
    "attention": "bhid,bhjd->bhij",
    "batch_matmul2": "bhij,bhjd->bhid",
    "batch_matmul3": "bxid,bjd->bxij",
    "batch_matmul4": "bxij,bjd->bxid",
    "batch_matrix_vector_multiply": "bid,bijd->bij",
    "alphaflod_usecase1": "hij,ijc->ihc",
    "alphaflod_usecase2": "rac,rab->rbc",
    "alphaflod_usecase3": "ra,rab->rb",
    "alphaflod_usecase4": "qhc,khc->qkh",
    "alphaflod_usecase5": "nm,mrc->nrc",
    "alphaflod_usecase6": "abc,adc->bdc",
    "alphaflod_usecase7": "dceb,cef->dbf",
    "alphaflod_usecase8": "acb,ade->dceb",
    "alphaflod_usecase9": "qkc,ch->hqk",
    "alphaflod_usecase10": "bhqk,bkhc->bqhc",
    "alphaflod_usecase11": "bqa,ahc->bqhc",
}

if args.einsum_testcase not in usecase_to_equations:
    print("not supported test case.")
    print(f"please choose cases in: {list(usecase_to_equations.keys())}")
    exit(0)

equation = usecase_to_equations[args.einsum_testcase]


def get_input_tensors_shapes(equation):
    input_eqs = equation.split("->")[0].split(",")
    shapes = []
    meet = {}
    for eq in input_eqs:
        shape = []
        # deal with batch_permute, not general
        if "..." in eq:
            eq = eq.split("...")[1]
            for d in range(args.ellipsis_dims):
                shape.append(
                    random.randint(
                        args.min_size_for_each_dim, args.max_size_for_each_dim
                    )
                )

        for s in eq:
            if s in meet:
                shape.append(meet[s])
            else:
                random_shape = random.randint(
                    args.min_size_for_each_dim, args.max_size_for_each_dim
                )
                shape.append(random_shape)
                meet[s] = random_shape
        shapes.append(shape)
    return shapes


input_shapes = get_input_tensors_shapes(equation)

print(f"\n# {args.einsum_testcase}")
print(f'einsum equation: "{equation}"')
shapes_str = ", ".join([f"{s}" for s in input_shapes])
print(f"input tensor shapes: {shapes_str}")

# oneflow
input_tensors = [flow.rand(*shape, device="cuda") for shape in input_shapes]
for i in range(args.warm_up_iters + args.run_iters):
    if i == args.warm_up_iters:
        start_time = time.time()
    flow._oneflow_internal.profiler.RangePush('do einsum then to numpy')
    out = flow.einsum(equation, *input_tensors)
    out_np = out.numpy()
    flow._oneflow_internal.profiler.RangePop()
total_time_ms = (time.time() - start_time) * 1000
oneflow_time_per_run_ms = total_time_ms / args.run_iters
print(
    f"oneflow avg time per iter: {oneflow_time_per_run_ms:.2f} ms (= {total_time_ms:.1f}ms / {args.run_iters})"
)

