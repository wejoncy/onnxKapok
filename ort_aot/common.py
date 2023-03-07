import numpy
from collections import defaultdict, OrderedDict
import onnx
import copy
import enum
import numpy as np
import symbolic_shape_infer
from abc import ABCMeta, abstractmethod
from logger import logger

_ENABLE_PARALLEL_COMPILE = False


class DataType(enum.Enum):
    UNDEFINED = 0
    # Basic types.
    FLOAT = 1   # float
    UINT8 = 2   # uint8_t
    INT8 = 3    # int8_t
    UINT16 = 4  # uint16_t
    INT16 = 5   # int16_t
    INT32 = 6   # int32_t
    INT64 = 7   # int64_t
    STRING = 8  # string
    BOOL = 9    # bool
    # Advanced types
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14     # complex with float32 real and imaginary components
    COMPLEX128 = 15    # complex with float64 real and imaginary components
    # Future extensions go here.


class AttributeType(enum.Enum):
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    SPARSE_TENSOR = 11
    TYPE_PROTO = 13

    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSORS = 12
    TYPE_PROTOS = 14


# This map is used for converting TensorProto values into Numpy arrays
TENSOR_TYPE_TO_NP_TYPE = {
    int(onnx.TensorProto.FLOAT): np.dtype('float32'),
    int(onnx.TensorProto.UINT8): np.dtype('uint8'),
    int(onnx.TensorProto.INT8): np.dtype('int8'),
    int(onnx.TensorProto.UINT16): np.dtype('uint16'),
    int(onnx.TensorProto.INT16): np.dtype('int16'),
    int(onnx.TensorProto.INT32): np.dtype('int32'),
    int(onnx.TensorProto.INT64): np.dtype('int64'),
    int(onnx.TensorProto.BOOL): np.dtype('bool'),
    int(onnx.TensorProto.FLOAT16): np.dtype('float16'),
    # Native numpy does not support bfloat16 so now use float32 for bf16 values
    int(onnx.TensorProto.BFLOAT16): np.dtype('float32'),
    int(onnx.TensorProto.DOUBLE): np.dtype('float64'),
    int(onnx.TensorProto.COMPLEX64): np.dtype('complex64'),
    int(onnx.TensorProto.COMPLEX128): np.dtype('complex128'),
    int(onnx.TensorProto.UINT32): np.dtype('uint32'),
    int(onnx.TensorProto.UINT64): np.dtype('uint64'),
    int(onnx.TensorProto.STRING): np.dtype('object')
}

# Currently native numpy does not support bfloat16 so TensorProto.BFLOAT16 is ignored for now
# Numpy float32 array is only reversed to TensorProto.FLOAT
NP_TYPE_TO_TENSOR_TYPE = {
    v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items() if k != onnx.TensorProto.BFLOAT16}

NP_TYPE_C_TYPE = {numpy.bool_: 'bool',
                  numpy.byte: 'int8_t',
                  numpy.ubyte: 'uint8_t',
                  numpy.short: 'short',
                  numpy.ushort: 'unsigned short',
                  numpy.intc: 'int',
                  numpy.uintc: 'unsigned int',
                  numpy.int_: 'long',
                  numpy.int64: 'int64_t',
                  numpy.uint: 'unsigned long',
                  numpy.longlong: 'long long',
                  numpy.ulonglong: 'unsigned long long',
                  numpy.single: 'float',
                  numpy.float32: 'float',
                  numpy.double: 'double',
                  numpy.longdouble: 'long double',
                  }


class NodeVisitor(object):
    def __init__(self):
        pass

    # interface for codegen/lowering
    @abstractmethod
    def visit(self, node, context, indent: int):
        pass


class CodeGenContext(object):
    def __init__(self, var_map: dict):
        self.var_map = var_map
        self.vectorized_var_set: Dict = set()


class SpecialVar(object):
    def __init__(self):
        self.input_args = "input_args"
        self.output_args = "output_args"
        self.input_args_size = "input_args_size"
        self.parallel_loop_start = "p_loop_start"
        self.parallel_loop_end = "p_loop_end"
        self.dynamic_shape_args = "dynamic_shape_args"
        self.rbase = "rbase"


def add_all_intermidiate_values(model):
    model_proto = copy.deepcopy(model)
    # model_proto, check = simplify(model_proto)
    org_outputs = [x.name for x in model_proto.graph.output]
    for node in model_proto.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model_proto.graph.output.extend([onnx.ValueInfoProto(name=output)])
    return model_proto


def get_symbol_shape(model_path):
    if isinstance(model_path, str):
        model = onnx.load(model_path)
    else:
        model = model_path
    symbolic_shape_infer.logger.setLevel(symbolic_shape_infer.logging.ERROR)
    symbol_shape = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
        model, 2**31 - 1, True, guess_output_rank=True, verbose=1
    )

    return symbol_shape


def parse_onnx_to_numpyarray(value):
    data = None
    if isinstance(value, onnx.onnx_ml_pb2.NodeProto):
        assert len(value.attribute) == 1
        Attr_type = AttributeType(value.attribute[0].type)
        if Attr_type == AttributeType.INT:
            data = np.array([value.attribute[0].i])
        elif Attr_type == AttributeType.TENSOR:
            value = value.attribute[0].t
        elif Attr_type == AttributeType.FLOAT:
            data = np.array([value.attribute[0].f])
        else:
            raise Exception("not support")
    # handle      AttributeType.TENSOR
    if isinstance(value, onnx.onnx_ml_pb2.TensorProto):
        assert data is None
        data = onnx.numpy_helper.to_array(value)
    elif data is not None:
        pass
    else:
        assert RuntimeError("not support proto type")
    return data


class OnnxInGraph(object):
    def __init__(self, onnx_model: onnx.ModelProto):
        self.model_proto = onnx_model
        self.graph = self.model_proto.graph

        self.node_name2module = dict()
        self.produced_by = dict()
        self.consumed_by = dict()
        self.graph_input_names = []
        self.graph_output_names = []
        self.initializer_name2module = dict()
        self.tensor_type_shape_info = dict()
        self.value_info_map = dict()
        #####

    @staticmethod
    def get_all_shape_from_onnx_model(model):
        try:
            symbol_shape = get_symbol_shape(model)
        except Exception as e:
            print(e)
            symbol_shape = model
        if symbol_shape is not None:
            type_shape_dict = dict()
            for value_info in symbol_shape.graph.value_info:
                dim = value_info.type.tensor_type.shape.dim
                type_shape_dict[value_info.name] = (value_info.type.tensor_type.elem_type, [
                                                    d.dim_value or d.dim_param for d in dim],)

            for inp in model.graph.input:
                dim = inp.type.tensor_type.shape.dim
                assert inp.name not in type_shape_dict, ("repeat input name: %s" % inp.name)
                type_shape_dict[inp.name] = (inp.type.tensor_type.elem_type, [d.dim_value or d.dim_param for d in dim],)

            return type_shape_dict, {}
        import transformers
        import onnxruntime as ort

        model_proto = add_all_intermidiate_values(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            # "distilbert-base-uncased"
            "lordtt13/emo-mobilebert"
        )
        inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
        # inputs["input_mask"] = inputs["attention_mask"]
        # del inputs["attention_mask"]
        sess = ort.InferenceSession(path_or_bytes=model_proto.SerializeToString())
        if sess.get_inputs()[0].name == "input_ids":
            ret = sess.run([i.name for i in sess.get_outputs()], dict(inputs))
            runtime_shape = {key.name: ret[idx].shape for idx, key in enumerate(sess.get_outputs())}
        else:
            runtime_shape = {}

        return type_shape_dict, runtime_shape

    def gen_name2module_map(self, infer_shape=True):
        self.initializer_name2module.clear()
        self.value_info_map.clear()
        self.node_name2module.clear()
        self.produced_by.clear()
        self.consumed_by.clear()
        self.graph_input_names.clear()
        self.graph_output_names.clear()
        self.graph_input_names.clear()
        self.graph_input_names.clear()

        for vi in self.graph.value_info:
            self.value_info_map[vi.name] = vi
        # initializer name => initializer
        for initializer in self.graph.initializer:
            self.initializer_name2module[initializer.name] = initializer

        # node name => node
        node_idx = 0
        for node in self.graph.node:
            if node.name == "":
                node.name = str(node.op_type) + str(node_idx)

            node_idx += 1
            self.node_name2module[node.name] = node
            for out in node.output:
                # if node.op_type == "Constant":
                #    continue
                assert out not in self.produced_by, (
                    "multiple producers for node: %s" % out
                )
                self.produced_by[out] = [node]
            for inp in node.input:
                if inp in self.initializer_name2module:
                    continue
                if inp not in self.consumed_by:
                    self.consumed_by[inp] = []
                self.consumed_by[inp].append(node)

        for inp in self.graph.input:
            self.node_name2module["out_" + inp.name] = inp
        self.graph_input_names.extend(
            ["out_" + inp.name for inp in self.graph.input])

        for out in self.graph.output:
            self.node_name2module[
                "out_" + out.name
            ] = out  # add `out_` in case the output has the same name with the last node
        self.graph_output_names = [
            "out_" + out.name for out in self.graph.output]
        if infer_shape:
            symbol_shape, rt_shape = self.get_all_shape_from_onnx_model(self.model_proto)
            self.tensor_type_shape_info = symbol_shape
            for initializer in self.graph.initializer:
                self.tensor_type_shape_info[initializer.name] = (initializer.data_type, list(initializer.dims))
            self.rt_shape = rt_shape


class GraphIOBuffer(object):
    def __init__(self):
        self.var_buffer_in = []
        self.var_buffer_out = []
        self.const_buffer = []


class IndexSubGraph(object):
    def __init__(self):
        self.sub_graph_nodes = []
        self.input_name_exclude_constant = []
        self.input_name_ref_c = OrderedDict()
        self.output_name_ref_c = OrderedDict()
        self.reduce_op = []

    def analyze_input_output(self, tensor_type_shape_info, is_constant_func):
        for ink in list(self.input_name_ref_c.keys()):
            v = self.input_name_ref_c[ink]
            if ink in self.output_name_ref_c:
                self.input_name_ref_c.pop(ink)
                if self.output_name_ref_c[ink] <= v:
                    self.output_name_ref_c.pop(ink)

        mut_input_name = []
        for ink, count in self.input_name_ref_c.items():
            if not is_constant_func(ink):
                mut_input_name.append(ink)

        # re order inputs/output by dtype
        non_float_idx = []
        for idx, ink in enumerate(mut_input_name):
            if tensor_type_shape_info[ink][0] != 1:
                non_float_idx.append(idx)
        if len(non_float_idx) > 1:
            logger.info(
                f"subgraph has multiple non-float32 inputs, skip. {len(self.sub_graph_nodes)} nodes was skipped.")
            return
        if not non_float_idx or non_float_idx[0] == 1 or len(mut_input_name) == 1:
            pass
        else:
            mut_input_name[1], mut_input_name[non_float_idx[0]] = mut_input_name[non_float_idx[0]], mut_input_name[1]
        self.input_name_exclude_constant = mut_input_name


class HardwareContext(object):
    def __init__(self, device_id, vec_lanes):
        self.vec_lanes = vec_lanes
        self.device_id = device_id
