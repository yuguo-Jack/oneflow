import oneflow as flow
from oneflow.framework.tensor_tuple_util import get_tensor_tuple_converter_pair

def trace(func):
    convert_from_tensor_tuple = None
    def traced(*args):
        inputs = flow._oneflow_internal.TensorTuple(args)
        jit_function = flow._oneflow_internal.jit.create_or_find_function(inputs)
        assert jit_function.has_input_placeholders()
        if not jit_function.has_recorded_instructions():
            try:
                flow._oneflow_internal.jit.instructions_record_begin()
                ret = func(inputs)
            finally:
                flow._oneflow_internal.jit.instructions_record_end()
            flow._oneflow_internal.jit.instructions_record_move_to(jit_function)
            assert jit_function.has_recorded_instructions()
            assert not jit_function.has_output_placeholders()
            nonlocal convert_from_tensor_tuple
            convert, convert_from_tensor_tuple = get_tensor_tuple_converter_pair(ret)
            jit_function.set_output_placeholders(convert(ret))
        return convert_from_tensor_tuple(jit_function(inputs))
    return traced
