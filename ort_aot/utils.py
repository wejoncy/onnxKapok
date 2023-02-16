
import onnx
from symbolic_shape_infer import get_shape_from_value_info, get_elem_type_from_type_proto
from ir import ComputeBuffer
import sympy_utils
import common


from typing import List,Union


def convert_onnx_value_to_computebuffer(tensors: Union[onnx.ValueInfoProto, List[onnx.ValueInfoProto]], prefix=''):
    not_list= False
    if not isinstance(tensors, list):
        not_list = True
        tensors = [tensors]
    bufs = []
    for tensor in tensors:
        dtype = common.TENSOR_TYPE_TO_NP_TYPE[get_elem_type_from_type_proto(
            tensor.type)]
        shape = get_shape_from_value_info(tensor)
        shape = sympy_utils.sympy_symbol(shape)
        bufs.append(ComputeBuffer(prefix+tensor.name, dtype, shape))
    return bufs if not_list == False else bufs[0]
