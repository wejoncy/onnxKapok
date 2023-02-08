import common
import numpy as np
from abc import ABCMeta, abstractmethod
from enum import Enum
from collections import defaultdict, deque, OrderedDict
from typing import Union, List, Tuple, Dict, Set
import sympy_utils
import sympy
from sympy.codegen.rewriting import create_expand_pow_optimization
import copy
import re

import onnx
import onnx.numpy_helper

import node_sets
import de_compose


class IoConnection(object):
    def __init__(self):
        self.users = []
        self.producers = []


class Node(object):
    def __init__(self, node, input_nodes=None, input_constant=None, output_nodes=None):
        self.current_node = node
        self.input_nodes = input_nodes if input_nodes else []
        self.input_constant = input_constant if input_constant else []
        self.output_nodes = output_nodes if output_nodes else []
        self.output_shapes = OrderedDict()
        self.input_shapes = OrderedDict()

    @property
    def op_type(self):
        if isinstance(self.current_node, onnx.onnx_ml_pb2.ValueInfoProto):
            return "PlaceHolder"
        return self.current_node.op_type

    @property
    def name(self):
        return self.current_node.name

    def __eq__(self, other):
        try:
            self.current_node.HasField("op_type")
        except:
            return False

        return self.current_node.op_type == other

    def __hash__(self):
        return hash(self.current_node.name)


class ConnectionGraph(object):
    def __init__(self, model):
        self.model = model
        self.node_collection = []
        self.node_2_gnode = {}
        self.in_dgree = defaultdict(lambda: 0)
        self.entry_nodes = []
        self.constant_nodes = OrderedDict()
        self.type_and_shape = OrderedDict()
        self.decompose_dispatcher = de_compose.DecomposeDispatch()

    def get_or_create_gnode(self, node):
        if node.name not in self.node_2_gnode:
            self.node_2_gnode[node.name] = Node(node)
            if isinstance(node, onnx.NodeProto):
                self.node_2_gnode[node.name].input_shapes = {
                    inp: self.type_and_shape[inp]
                    for inp in node.input
                    if inp in self.type_and_shape
                }
                self.node_2_gnode[node.name].output_shapes = {
                    inp: self.type_and_shape[inp] for inp in node.output
                }

        return self.node_2_gnode[node.name]

    def try_decompose(self, node) -> (list):
        if node in node_sets.ReduceNodeSetInternal():
            return self.decompose_dispatcher(node)

        return []

    def decompose(self, graph):
        replace_nodes = {}
        for node in graph.node:
            r_nodes = self.try_decompose(node)
            if r_nodes:
                replace_nodes[node.name] = (node, r_nodes)
        for node, r_nodes in replace_nodes.values():
            graph.node.remove(node)
            graph.node.extend(r_nodes)

    def build_relationship(self):
        self.egraph = common.OnnxInGraph(self.model)
        self.decompose(self.egraph.graph)
        self.egraph.gen_name2module_map()
        egraph = self.egraph
        self.type_and_shape = egraph.tensor_type_shape_info
        self.constant_nodes.update(egraph.initializer_name2module)
        for node in egraph.graph.node:
            if node.op_type == "Constant":
                self.constant_nodes[node.name] = node
                continue
            gnode: Node = self.get_or_create_gnode(node)
            self.node_collection.append(gnode)
            for inp in node.input:
                if inp in egraph.produced_by:
                    in_gnode = self.get_or_create_gnode(egraph.produced_by[inp][0])
                    gnode.input_nodes.append(in_gnode)
                elif inp in egraph.initializer_name2module:
                    in_gnode = self.get_or_create_gnode(
                        egraph.initializer_name2module[inp]
                    )
                    gnode.input_constant.append(in_gnode)
                elif inp in egraph.graph_input_names:
                    in_gnode = self.get_or_create_gnode(egraph.node_name2module[inp])
                    in_gnode.output_nodes.append(gnode)
                    self.in_dgree[gnode] += 1
                    gnode.input_nodes.append(in_gnode)
                    self.entry_nodes.append(in_gnode)
                else:
                    raise Exception("input not found")

            for out in node.output:
                if out in egraph.consumed_by:
                    for next_node in egraph.consumed_by[out]:
                        out_gnode = self.get_or_create_gnode(next_node)
                        gnode.output_nodes.append(out_gnode)
                        self.in_dgree[out_gnode] += 1
                elif "out_" + out in egraph.graph_output_names:
                    out_gnode = self.get_or_create_gnode(
                        egraph.node_name2module["out_" + out]
                    )
                    out_gnode.input_nodes.append(gnode)
                    gnode.output_nodes.append(out_gnode)
                    self.in_dgree[out_gnode] += 1
                else:
                    raise Exception("output not found")
        # for inp in egraph.graph_input_names:
        #    in_gnode:Node = self.get_or_create_gnode(egraph.node_name2module[inp])
        #    len(in_gnode.output_nodes) == 2
        #    [i.name for i in in_gnode.output_nodes]


class ComputeBuffer(object):
    def __init__(
        self, name: str, type: int = 0, shape: list = None, data: np.ndarray = None
    ):
        self.name = name
        self.type = type
        self.shape = self.parse_shape(shape)
        self.data = data
        self.loop_index: List[sympy.Expr] = None

    def parse_shape(self, shape: list):
        if shape is None:
            return None
        else:
            symbol_shapes = []
            for x in shape:
                if isinstance(x, str):
                    symbol_shape = sympy.Symbol(x)
                elif isinstance(x, sympy.Symbol):
                    symbol_shape = x
                else:
                    symbol_shape = sympy.Integer(x)
                symbol_shapes.append(symbol_shape)
            return symbol_shapes

    def __eq__(self, o):
        if isinstance(o, ComputeBuffer):
            return self.name == o.name
        elif isinstance(o, str):
            return self.name == o
        else:
            raise Exception("not supported")

    def __hash__(self):
        return hash(self.name)


class CodeGenContext(object):
    def __init__(self, var_map: dict):
        self.var_map = var_map
        self.vectorized_var_set: Dict = set()


class IRNode:
    def __init__(self):
        self.parent = None
        self.vectorization = False
        self.input: List[ComputeBuffer] = []
        self.output: List[ComputeBuffer] = []

    @abstractmethod
    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        pass


class LoopAttr(Enum):
    ScalarLoop = 0
    Parallel = 1
    Reduce = 2
    Vectorization = 3


class Loop(IRNode):
    def __init__(self):
        super().__init__()
        self.var: sympy.Expr = None
        self.start = sympy.Integer(0)
        self.end = sympy.Integer(0)
        self.step = sympy.Integer(1)
        self.body = None
        self.depth = 0
        self.parallel: bool = False
        self.parallel_nest_loop: Loop = None
        self.attributes = LoopAttr.ScalarLoop
        self.forward_var_set: Dict[ComputeBuffer] = OrderedDict()
        self.var_need_post_process: OrderedDict = {}

    def visit(self, var):
        if isinstance(var, sympy.Mul):
            return f"({self.visit(var.args[0])} * {self.visit(var.args[1])})"
        elif isinstance(var, sympy.Add):
            return f"({self.visit(var.args[0])} + {self.visit(var.args[1])})"
        elif isinstance(var, sympy_utils.FloorDiv):
            return f"({self.visit(var.args[0])} / {self.visit(var.args[1])})"
        else:
            return str(var)

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        var_map = var_context.var_map
        vec_var_set = var_context.vectorized_var_set
        need_indent = " " * indent
        dec_header = ""
        # forward declaration
        for fvar, buffer in self.forward_var_set.items():
            str_var = str(fvar)
            if buffer.shape is not None and buffer.shape[-1] == 1:
                dec_header += need_indent + f"float {var_map[str_var]} = 0.0f;\n"
                if self.vectorization:
                    dec_header += (
                        need_indent
                        + f"mipp::Reg<float> vec_{var_map[str_var]} = 0.0f;\n"
                    )
                    vec_var_set.add(var_map[str_var])
                    self.var_need_post_process[str_var] = f"vec_{var_map[str_var]}"
            else:
                assert False, "buffer should be defined in the execution-block"
                dec_header += (
                    need_indent
                    + f"float e_{var_map[str_var]}[{buffer.shape[-1]}] __attribute__((aligned(64))) = {{0.0}};\n"
                )

        ##forward declare vectorization vars
        v_step = ""
        if self.vectorization:
            v_step = "mipp::N<float>()"
        src = dec_header
        src += need_indent + f"//@{self.attributes.name}\n"
        src += need_indent
        if self.parallel:
            p_var = f"{self.var}"
            p_var += (
                f"_{self.parallel_nest_loop.var}" if self.parallel_nest_loop else ""
            )
            src += f"for (int {p_var}={common.SpecialVar().parallel_loop_start}; {p_var}<{common.SpecialVar().parallel_loop_end}; {p_var}+={self.step}){{\n"
            if self.parallel_nest_loop:
                src += (
                    need_indent
                    + f"auto {self.var} = {p_var}/{self.parallel_nest_loop.end};\n"
                )
                nest_var = self.parallel_nest_loop.var
                src += (
                    need_indent
                    + f"auto {nest_var} = {p_var}%{self.parallel_nest_loop.end};\n"
                )
        else:
            # if self.depth == 0 and self.start == 0:
            #    if not self.end.is_Number:
            #        src += f'#pragma omp assume holds({self.end} % 16 == 0 && {self.end} > 0)\n'+need_indent
            #    src += f'#pragma omp simd\n'+need_indent
            #    src += f'#pragma vector aligned\n'+need_indent
            src += f"for (int {self.var}={self.visit(self.start)}; {self.var}<{self.visit(self.end)}; \
{self.var}+={v_step if self.vectorization else self.step}){{\n"

        if isinstance(self.body, list):
            for idx, g in enumerate(self.body):
                src += g.code_gen(var_context, indent + 4)
        else:
            src += self.body.code_gen(var_context, indent + 4)
        src = src + need_indent + "}\n"
        return src


# Generally, we use it to handle vectorization type change (vec->scalar)
class PostProcessBlock(Loop):
    def __init__(self, loop: Loop):
        super().__init__()
        self.global_connections: Dict[str, IoConnection] = None
        self.body: List[loop] = [loop]
        self.op_map: Dict[str, str] = {
            "ReduceMax": "hmax",
            "ReduceMin": "hmin",
            "ReduceSum": "hadd",
        }

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        var_map = var_context.var_map
        need_indent = " " * indent
        to_be_handled_vars_map = self.body[0].var_need_post_process
        src = ""
        for s_var, v_var in to_be_handled_vars_map.items():
            # TODO determinate the type of Op, ReduceMax or ReduceMin or ReduceSum etc
            # hmin hmax hadd
            assert (
                s_var in var_map and s_var in self.global_connections
            ), f"{s_var} not in var_map"
            vec_func = self.op_map[self.global_connections[s_var].producers[0].op_type]
            w_var = var_map[s_var]
            if vec_func == "hadd":
                src += need_indent + f"{w_var} = {w_var} + mipp::{vec_func}({v_var});\n"
            elif vec_func == "hmax":
                src += (
                    need_indent
                    + f"{w_var} = std::max({w_var} , mipp::{vec_func}({v_var}));\n"
                )
            else:
                raise NotImplementedError(f"not support {vec_func} yet")
            src += need_indent + f"{v_var} = {w_var};\n"
        return src


class FunctionNode(IRNode):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.input = inputs
        self.output = outputs
        self.name: str = ""
        self.const_var = []
        self.shape_var = []
        self.body: List[ExecutionBlock] = None
        self.hw_context: common.HardwareContext = None

    def lower(self):
        assert len(self.body) == 1, "multiple body not supported in function node"
        shape_var = [i for i in self.body[0].shape if i.is_symbol]
        self.shape_var = list(set(shape_var))
        self.shape_var.sort(key=shape_var.index)

        self.body[0].gen_var(self.const_var)
        self.body[0].analyze_io_connections()

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        if not var_context:
            var_map = self.body[0].var_map
        # self.output[0].type.tensor_type.shape.dim[1]

        # in_param = [f"const float* e_{var_map[i.name]}" for i in self.input]

        in_param = [
            f"const float** {common.SpecialVar().input_args}",
            f"int {common.SpecialVar().input_args_size}",
        ]
        if self.body[0].body.parallel:
            in_param += [
                f"int64_t {common.SpecialVar().parallel_loop_start}, int64_t {common.SpecialVar().parallel_loop_end}"
            ]
        in_param += [f"const int64_t {i}" for i in self.shape_var]
        in_param = ",".join(in_param)

        # out_param = ",".join([f"float* e_{var_map[i.name]}" for i in self.output])
        out_param = f"float** output_args"

        func_signature = f"int {self.name}({in_param}, {out_param}) {{\n"

        code = ""
        code += func_signature
        indent += 4
        need_indent = " " * indent
        # DEBUG code
        assert_code = f"""
    #ifdef DEBUG_
    if ({common.SpecialVar().input_args_size} != {len(self.input)}){{
        printf(" assert {common.SpecialVar().input_args_size} != {len(self.input)} failed, please check your model or code\\n\\n");
        abort();
    }}
    #endif
"""
        assert self.hw_context is not None
        bytes_lanes = self.hw_context.vec_lanes * 4  # (sizeof(float)

        code += assert_code
        restrict = f"__restrict__  __attribute__((aligned ({bytes_lanes})))"
        parse_input = [
            need_indent
            + f"const float* {restrict} e_{var_map[i.name]} = {common.SpecialVar().input_args}[{idx}];"
            for idx, i in enumerate(self.input)
        ]
        code += "\n".join(parse_input) + "\n\n"

        parse_output = [
            need_indent
            + f"float* {restrict} e_{var_map[i.name]} = {common.SpecialVar().output_args}[{idx}];"
            for idx, i in enumerate(self.output)
        ]
        code += "\n".join(parse_output) + "\n\n"

        self.body[0].input
        de_composed_const_var = {}
        for const in self.const_var:
            if isinstance(const, onnx.NodeProto):
                if const.attribute[0].type == 4:
                    for i in const.attribute:
                        de_composed_const_var[const.output[0]] = i.t
                else:
                    de_composed_const_var[const.output[0]] = const
            else:
                de_composed_const_var[const.name] = const

        for name, const in de_composed_const_var.items():
            if isinstance(const, onnx.TensorProto):
                np_array = onnx.numpy_helper.to_array(const).reshape(-1)
                np_array_i32 = np_array.view(np.int32)
                # x_arrstr = np.char.mod("%.6ef", np_array)
                x_arrstr = np.char.mod("%#x", np_array_i32)
                x_str = ",".join(x_arrstr)
                if np_array.size == 1:
                    continue
                    # const_declare = (
                    #    f"static constexpr float e_{var_map[name]} = {x_str};\n"
                    # )
                else:
                    const_declare = (
                        f"static constexpr int32_t e_{var_map[name]}_i32p[] __attribute__((aligned({bytes_lanes}))) = {{{x_str}}};\n"
                        + need_indent
                        + f"static const float* e_{var_map[name]} = (float*)e_{var_map[name]}_i32p;\n"
                    )
            elif isinstance(const, onnx.NodeProto):
                if const.attribute[0].type == 2:  # int
                    v = const.attribute[0].i
                    const_declare = f"static constexpr int e_{var_map[name]} = {v};\n"
                elif const.attribute[0].type == 1:
                    v = const.attribute[0].f
                    const_declare = (
                        f"static constexpr float e_{var_map[name]} = {v}f;\n"
                    )
                else:
                    raise Exception("not supported")
            else:
                raise Exception("not supported")
            code += need_indent + const_declare

        self.body[0].hw_context = self.hw_context

        code += self.body[0].code_gen(None, indent)
        code += need_indent + "return 12;\n"
        code += "}\n"

        return code


class ModuleNode(IRNode):
    def __init__(self):
        super().__init__()
        self.body: List[FunctionNode] = []
        self.has_vectorization = False

    def lower(self, function_recipes: list, func: callable):
        allow_vectorize = True
        for function_recipe in function_recipes:
            function: FunctionNode = func(
                *function_recipe, allow_vectorize=allow_vectorize
            )
            self.body.append(function)
        self.has_vectorization = allow_vectorize

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        code = """
#include <cmath>
"""
        if self.has_vectorization:
            code += """
#include <mipp/mipp.h>
using namespace mipp;
"""
        code += """
extern "C"{
"""

        for idx, func in enumerate(self.body):
            code += f"//the {idx}th function/sub_graph\n"
            code += func.code_gen(None, indent)
        # extern C
        code += "}\n"
        return code


class ComputeNode(IRNode):
    def __init__(self, op_type, inputs, outputs, op_name: str = ""):
        super().__init__()
        self.op_type_ = op_type
        self.input = list(inputs)
        self.output = list(outputs)
        self.op_name = op_name

    @property
    def op_type(self):
        return self.op_type_

    def gen_cpp_code_for_op(self, var_context: CodeGenContext):
        var_map = var_context.var_map
        vec_var_map = var_context.vectorized_var_set
        ori_named_vars_i = [var_map[i] for i in self.input]
        ori_named_vars_o = [var_map[i] for i in self.output]
        suffix = ["", ""]

        named_vars_i = ori_named_vars_i.copy()
        named_vars_o = ori_named_vars_o.copy()

        for i in range(len(named_vars_i)):
            # if named_vars_i[i] is constant scalar, just use it
            if named_vars_i[i] in var_map:
                named_vars_i[i] = var_map[named_vars_i[i]][0]
                # without suffix 'f', compiler will see it as double
                suffix[i] = "f"

        vectorized_prefix = "std::"
        # use it to check if the input is a vector
        # for vectorized op, we need to add "vec_" for reduced output vars
        # in tile_loop, we split loops into main loop and tail loop
        # main loop is vectorized, tail loop is not.
        # so we have two kinds of output vars
        is_input_1_vec = False
        if self.vectorization:
            vectorized_prefix = "mipp::"
            for idx, var in enumerate(named_vars_i):
                if str(var) in vec_var_map:
                    named_vars_i[idx] = f"vec_{var}"
                    is_input_1_vec = idx == 1
        # canonize mul as add
        if self.op_type == "Sub" and (
            not isinstance(named_vars_i[0], str) or is_input_1_vec
        ):
            named_vars_i[1] = "-" + named_vars_i[1]
            self.op_type_ = "Add"
        if (
            self.op_type == "Div"
            and is_input_1_vec
            and not named_vars_i[0].startswith("vec_")
        ):
            named_vars_i[0] = f"mipp::Reg<float>({named_vars_i[0]})"
        # always trying to put the constant to the right
        if (suffix[0] == "f" or is_input_1_vec) and self.op_type in ["Add", "Mul"]:
            suffix[0], suffix[1] = suffix[1], suffix[0]
            named_vars_i[0], named_vars_i[1] = named_vars_i[1], named_vars_i[0]

        named_vars_i[0] = f"{named_vars_i[0]}{suffix[0]}"

        if len(named_vars_i) == 2:
            raw_named_vars_1 = named_vars_i[1]
            named_vars_i[1] = f"{named_vars_i[1]}{suffix[1]}"
            # ",".join(np.char.mod("%.6ef", var_map[named_vars_i[i]]))
        assert len(named_vars_i) in [1, 2, 3]
        assert len(named_vars_o) == 1

        named_var_o = named_vars_o[0]
        src = "auto "
        if self.op_type == "Add":
            src += f"{named_var_o} = {named_vars_i[0]} + ({named_vars_i[1]});\n"
        elif self.op_type == "Sub":
            src += f"{named_var_o} = {named_vars_i[0]} - ({named_vars_i[1]});\n"
        elif self.op_type == "Div":
            src += f"{named_var_o} = {named_vars_i[0]} / ({named_vars_i[1]});\n"
        elif self.op_type == "Mul":
            src += f"{named_var_o} = {named_vars_i[0]} * ({named_vars_i[1]});\n"
        elif self.op_type == "Relu":
            src += f"{named_var_o} = {vectorized_prefix}max({named_vars_i[0]}, {'vec_zero' if self.vectorization else '0'});\n"
        elif self.op_type == "Fma":
            # a*b+c
            src += f"{named_var_o} = std::fma({named_vars_i[0]}, ({named_vars_i[1]},{named_vars_i[2]}));\n"
        elif self.op_type == "Pow":
            # rewrite pow as mul
            if raw_named_vars_1 == 2:
                src += f"{named_var_o} = {named_vars_i[0]} * {named_vars_i[0]};\n"
            elif raw_named_vars_1 == 3:
                src += f"{named_var_o} = {named_vars_i[0]} * {named_vars_i[0]}* {named_vars_i[0]};\n"
            else:
                src += f"{named_var_o} = pow({named_vars_i[0]},{named_vars_i[1]});\n"
        elif self.op_type == "Sqrt":
            src += f"{named_var_o} = {vectorized_prefix}sqrt({named_vars_i[0]});\n"
        # elif self.op_type == "Cast":
        #    src += f"{named_var_o} = sqrt({named_vars_i[0]});\n"
        elif self.op_type == "Erf":
            # 4/sqrt(M_PI) = 7.0898154036220635
            src += f"{named_var_o} = {vectorized_prefix}tanh({named_vars_i[0]}*({named_vars_i[0]}*{named_vars_i[0]}*0.044715f+0.5f))*7.0898154036220635f;\n"
        elif self.op_type == "Exp":
            src += f"{named_var_o} = {vectorized_prefix}exp({named_vars_i[0]});\n"
        else:
            raise Exception(f"not supported {self.op_type}")
        return src

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        space_indent = " " * indent
        src = space_indent + f"// {self.op_name} {self.op_type}\n"

        if self.op_type == "Relu":
            src += space_indent + "mipp::Reg<float> vec_zero = 0.0f;\n"
        src += space_indent + self.gen_cpp_code_for_op(var_context)
        return src


class ReduceNode(ComputeNode):
    def __init__(self, body: ComputeNode, axis=-1):
        super().__init__(body.op_type, body.input, body.output, body.op_name)
        self.axis = axis
        self.body: ComputeNode = body
        self.input = body.input
        self.output = body.output

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        var_map = var_context.var_map
        vec_var_map = var_context.vectorized_var_set
        code = "\n"
        try:
            input_1 = var_map[var_map[self.input[1]]]
        except:
            input_1 = np.NaN
            pass
        assert len(self.input) == 1 or (input_1 == -1).all()
        named_var_i = var_map[self.input[0]]
        named_var_o = var_map[self.output[0]]
        # this var is vectorized, add prefix 'vec_'
        vec_pre = "mipp::" if self.vectorization else "std::"
        if self.vectorization and named_var_o in vec_var_map:
            named_var_o = "vec_" + named_var_o
        if named_var_i != self.input[0]:
            code += " " * indent + f"// {named_var_i} = {self.input[0]};\n"
            code += " " * indent + f"// {named_var_o} = {self.output[0]};\n"
        if self.body.op_type == "ReduceSum":
            code += " " * indent + f"{named_var_o} = {named_var_o}+{named_var_i};\n"
        elif self.body.op_type == "ReduceMax":
            code += (
                " " * indent
                + f"{named_var_o} = {vec_pre}max({named_var_o},{named_var_i});\n"
            )
        else:
            raise Exception(f"not supported {self.body.op_type}")
        return code


class Indexer:
    def __init__(self):
        self.buf_index = None

    def cal_stride(self, shape):
        expand_opt = create_expand_pow_optimization(6)
        stride = [1]
        for i in range(len(shape) - 1, 0, -1):
            stride.append(expand_opt(stride[-1] * shape[i]))
        return stride[::-1]

    def code_gen(self, named_var: str, buf: ComputeBuffer):
        if buf.data is not None and buf.data.size == 1:
            return f"{named_var}"
        else:
            shape = buf.shape or (buf.data is not None and buf.data.shape) or [1]
            index_of_dim_1 = [i for i in range(len(shape)) if shape[i] == 1]
            stride = self.cal_stride(shape)
            index: sympy.Expr = buf.loop_index or [
                sympy_utils.sympy_symbol(f"i_{i}")
                for i in range(len(shape) - 1, -1, -1)
            ]
            if len(index) > len(shape):
                index = index[len(index) - len(shape) :]
            # broadcast handling
            br_index = [v for idx, v in enumerate(index) if idx not in index_of_dim_1]
            br_stride = [v for idx, v in enumerate(stride) if idx not in index_of_dim_1]

            res = sympy.Matrix([br_index]).dot(sympy.Matrix(br_stride))
            # res = res.subs(shape[0], 1)
            gs = re.findall("([a-zA-Z0-9_]+)\*\*(\d)", str(res))
            assert (
                gs == []
            )  # or gs[0][1] == '2', f"TODO fix me when pow {gs[0][1]} or other"
            # res= re.sub('([a-zA-Z0-9_]+)\*\*(\d)','\g<1>*\g<1>',str(res))
            return f"{named_var}[{res}]"
        pass


class LoadNode(IRNode):
    def __init__(self, buf: ComputeBuffer):  # ComputeBuffer
        super().__init__()
        self.input = buf
        self.to_buf = "to"

    @property
    def op_type(self):
        return "Load"

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        var_map = var_context.var_map
        vec_var_map = var_context.vectorized_var_set
        space_indent = " " * indent
        code = ""
        var_name = self.input.name
        assert var_name in var_map, f"name {var_name} not found in var_map"
        named_var = var_map[var_name]
        # if named_var is constant, no need to load
        if named_var in var_map:
            assert False, f"TODO: {named_var} is constant, no need to load"
            v = var_map[named_var]
            assert isinstance(v, (np.ndarray))
            if self.vectorization:
                vec_var_map.add(named_var)
                return (
                    code
                    + space_indent
                    + f"mipp::Reg<float> vec_{named_var} = {v[0]}f;\n"
                )
            else:
                return code + space_indent + f"auto {named_var} = {v[0]}f;\n"

        if named_var != var_name:
            code += space_indent + f"//load ... {var_name} = {named_var};\n"
        annotated_var = Indexer().code_gen(named_var, self.input)

        if self.vectorization:
            vec_var_map.add(named_var)
            return (
                code
                + space_indent
                + f"mipp::Reg<float> vec_{named_var} = &e_{annotated_var};\n"
            )
        else:
            return code + space_indent + f"auto {named_var} = e_{annotated_var};\n"


class StoreNode(IRNode):
    def __init__(self, buf: ComputeBuffer):  # ComputeBuffer
        super().__init__()
        self.to_var = buf

    @property
    def op_type(self):
        return "Store"

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        var_map = var_context.var_map
        code = ""
        space_indent = code + " " * indent
        var_name = self.to_var.name
        assert var_name in var_map
        named_var = var_map[var_name]

        if named_var != var_name:
            code += " " * indent + f"// store ....{var_name} = {named_var};\n"
        annotated_var = Indexer().code_gen(named_var, self.to_var)
        if self.vectorization:
            return space_indent + f"{named_var}.store(&e_{annotated_var});\n"
        else:
            return space_indent + f"e_{annotated_var} = {named_var};\n"


class ExecutionBlock(IRNode):
    def __init__(self, group: List[IRNode]):
        super().__init__()
        self.input: list[ComputeBuffer] = []
        self.output: list[ComputeBuffer] = []
        self.constant_vars: list[ComputeBuffer] = []
        self.intermediate_var = OrderedDict()
        self.load = OrderedDict()
        self.loop_stack = []
        self.has_reduce = False
        # TODO support multiple outputs
        self.type = list(group[0].output_shapes.values())[0][0]
        self.shape = self.extract_shape(group)
        self.var_map = OrderedDict()
        self.forward_var_set = [OrderedDict()]
        self.body = None
        self.hw_context = None
        self.connections: Dict[str, IoConnection] = OrderedDict()

        self.group = self.translate(group)
        self.fused_groups = [self.group]

    def extract_shape(self, group: List[IRNode]):
        assert len(group[-1].output_shapes) == 1
        if group[-1].current_node in node_sets.ReduceNodeSetInternal():
            shape = [
                sympy_utils.sympy_symbol(i)
                for i in list(group[-1].input_shapes.values())[0][1]
            ]
            self.has_reduce = True
        else:
            shape = [
                sympy_utils.sympy_symbol(i)
                for i in list(group[-1].output_shapes.values())[0][1]
            ]
        return shape

    def build_inner_most_loop(self):
        self.loop_stack.append(
            sympy_utils.sympy_symbol(f"i_{str(len(self.loop_stack))}")
        )
        loop = Loop()
        loop.var = self.loop_stack[-1]

        loop.start = sympy.Integer(0)
        loop.end = self.shape[-1]
        loop.body = self.group
        if self.has_reduce:
            loop.attributes = LoopAttr.Reduce
        else:
            loop.attributes = LoopAttr.Vectorization

        loop.depth = 0
        loop.forward_var_set = self.forward_var_set[0]

        return loop

    def build_loop(self):
        body = self.build_inner_most_loop()
        for i in range(len(self.shape) - 2, -1, -1):
            loop = Loop()
            self.loop_stack.append(f"i_{str(len(self.loop_stack))}")
            loop.var = self.loop_stack[-1]
            loop.start = 0
            loop.end = self.shape[i]
            loop.body = body
            loop.depth = len(self.loop_stack) - 1
            body = loop
        return body

    def gen_var(self, external_var):
        exist_var = set()

        def legal_name(name):
            nonlocal exist_var
            import re

            pos = name.rfind("/")
            if pos != -1:
                name = name[pos + 1 :]
            else:
                name = re.sub(r"[^a-zA-Z0-9]", "_", name)
                if len(name) > 20:
                    name = name[-20:]

            # 1. assure name is legal, not start with digit
            # 2. assure name is different from the original
            name = "aot_" + name
            while name in exist_var:
                name = name + "_1"

            exist_var.add(name)
            return name

        for forward_var_set in self.forward_var_set:
            for inp in forward_var_set:
                self.var_map[inp] = legal_name(inp)
        for inp in self.input:
            self.var_map[inp.name] = legal_name(inp.name)
        for out in self.output:
            self.var_map[out.name] = legal_name(out.name)
        for var in self.intermediate_var:
            self.var_map[var] = legal_name(var)
        for var in self.load:
            self.var_map[var] = legal_name(var)
            v = (
                self.load[var].data.reshape(-1)
                if self.load[var].data is not None
                else None
            )
            if v is not None and v.size == 1:
                v_v = self.var_map[var]
                assert v_v not in self.var_map
                self.var_map[v_v] = v

        for out in external_var:
            var = out.name
            self.var_map[var] = legal_name(var)
            if isinstance(out, onnx.NodeProto):
                v = onnx.numpy_helper.to_array(out.attribute[0].t).reshape(-1)
                assert v.size == 1, "only support scalar"
                v_v = self.var_map[var]
                self.var_map[v_v] = v

    def code_gen(self, var_context: CodeGenContext, indent: int = 0):
        assert not var_context
        var_context = CodeGenContext(self.var_map)
        var_map = var_context.var_map
        need_indent = " " * indent
        src = ""
        # forward declaration of intermediate buffer for sub-loop for better reuse
        bytes_lanes = self.hw_context.vec_lanes * 4
        dec_for_sub_loop = ""
        var_declared = set()
        if self.forward_var_set:
            for fvs in self.forward_var_set:
                for str_var, buffer in tuple(fvs.items()):
                    if buffer.shape[-1] == 1:
                        continue
                    if str_var in var_declared:
                        fvs.pop(str_var)
                        continue
                    var_declared.add(str_var)
                    dec_for_sub_loop += (
                        need_indent
                        + f"float e_{var_map[str_var]}[{buffer.shape[-1]}] __attribute__((aligned({bytes_lanes}))) = {{0.0}};\n"
                    )
                    fvs.pop(str_var)
        src += dec_for_sub_loop
        src += self.body.code_gen(var_context, indent)
        return src

    def lower(self):
        # assume shape [N, len, hidden]
        # add Loop()
        self.body = self.build_loop()

    def translate(self, group: List[Node]):
        new_group = []
        for g in group:
            node = g.current_node
            ir_node = ComputeNode(node.op_type, node.input, node.output, node.name)
            if node in node_sets.ReduceNodeSetInternal():
                self.has_reduce = True
                for o in node.output:
                    self.forward_var_set[0][o] = ComputeBuffer(
                        name=o, type=g.output_shapes[o][0], shape=g.output_shapes[o][1]
                    )
                new_group.append(ReduceNode(ir_node))
            else:
                new_group.append(ir_node)
        return new_group

    def insert_load_and_store(
        self, global_buffer: common.GraphIOBuffer, c_graph: ConnectionGraph
    ):
        input_name_map = {inp.name: inp for inp in self.input}
        output_name_map = {inp.name: inp for inp in self.output}
        new_group = []
        for g in self.group:
            for inp in g.input:
                producer_op = (
                    c_graph.egraph.produced_by[inp][0]
                    if inp in c_graph.egraph.produced_by
                    else None
                )

                if (
                    inp in self.load or inp in input_name_map
                ) and producer_op not in node_sets.ReduceNodeSetInternal():
                    load_buf = (
                        self.load[inp] if inp in self.load else None
                    ) or input_name_map[inp]
                    # we just skip unused load for constant scalar
                    if load_buf.data is not None and load_buf.data.size == 1:
                        continue
                    new_group.append(LoadNode(load_buf))
            new_group.append(g)
            for out in g.output:
                if out in output_name_map and not isinstance(g, ReduceNode):
                    new_group.append(StoreNode(output_name_map[out]))

        self.group = new_group

    def analyze_io_connections(self):
        for group in self.fused_groups:
            for g in group:
                ipt = [g.input] if not isinstance(g.input, list) else g.input

                for inp in ipt:
                    in_name = inp if isinstance(inp, str) else inp.name
                    if in_name not in self.connections:
                        self.connections[in_name] = IoConnection()
                    self.connections[in_name].users.append(g)

                for out in g.output:
                    out_name = out if isinstance(out, str) else out.name
                    if out_name not in self.connections:
                        self.connections[out_name] = IoConnection()
                    assert (
                        len(self.connections[out_name].producers) == 0
                    ), "multiple producers!!"
                    self.connections[out_name].producers.append(g)
        pass

    def analyze_io(
        self,
        global_buffer: common.GraphIOBuffer,
        c_graph: ConnectionGraph,
        cached_buffer: Dict[str, ComputeBuffer],
    ):
        inputs = defaultdict(lambda: 0)
        outputs = defaultdict(lambda: 0)
        loads = set()
        const_buffer_set = set([i.name for i in global_buffer.const_buffer])
        external_buffer_out_set = set([i.name for i in global_buffer.var_buffer_out])

        def is_const_input(inp):
            return inp in const_buffer_set or (
                inp in c_graph.egraph.produced_by
                and c_graph.egraph.produced_by[inp][0].op_type == "Constant"
            )

        for g in self.group:
            for inp in g.input:
                if not is_const_input(inp):
                    inputs[inp] += 1
                else:
                    loads.add(inp)
            for out in g.output:
                if out not in const_buffer_set:
                    outputs[out] = 1
                else:
                    raise Exception("const buffer can not be output")
        # self.intermediate_var  = inputs.intersection(outputs)

        for i in list(outputs.keys()):
            if i in list(inputs.keys()) and i not in external_buffer_out_set:
                outputs[i] = len(c_graph.egraph.consumed_by[i]) - inputs[i]
                self.intermediate_var[i] = 0
                inputs.pop(i)
                assert outputs[i] >= 0, "output buffer can not be input"
                if outputs[i] == 0:
                    outputs.pop(i)

        for v in outputs:
            assert v not in cached_buffer, "create buffer twice is not allowed"
            if "out_" + v in c_graph.egraph.graph_output_names:
                tensor: onnx.Tensor = c_graph.egraph.node_name2module[
                    "out_" + v
                ].type.tensor_type
                dims = tensor.shape.dim
                shape = [dim.dim_value or dim.dim_param for dim in dims]
                buffer = ComputeBuffer(v, tensor.elem_type, shape)
            else:
                type_and_shape = c_graph.egraph.tensor_type_shape_info[v]
                sp = [sympy_utils.sympy_symbol(type_and_shape[1][-1])]
                buffer = ComputeBuffer(v, type_and_shape[0], sp)
            cached_buffer[v] = buffer
            self.output.append(buffer)

        for v in inputs:
            if v in external_buffer_out_set:
                # print("intermediate value appears in output, skip load", v)
                continue
            if v in cached_buffer:
                type_and_shape = c_graph.egraph.tensor_type_shape_info[v]
                assert (
                    sympy_utils.sympy_symbol(type_and_shape[1][-1])
                    == cached_buffer[v].shape[-1]
                ), "???buffer not matched"
                buffer = cached_buffer[v]
            elif v in c_graph.egraph.graph_input_names:
                tensor: onnx.Tensor = c_graph.egraph.node_name2module[
                    v
                ].type.tensor_type
                dims = tensor.shape.dim
                shape = [dim.dim_value or dim.dim_param for dim in dims]
                # onnx.TensorProto.FLOAT
                buffer = ComputeBuffer(v, tensor.elem_type, shape)
                cached_buffer[v] = buffer
            else:
                buffer = cached_buffer[v]
            # elif v in c_graph.egraph.produced_by:
            #    tensor:onnx.Tensor = c_graph.egraph.produced_by[v][0]
            #    dims = tensor.shape.dim
            #    shape = [dim.dim_value or dim.dim_param for dim in dims]
            #    buffer = ComputeBuffer(v,tensor.elem_type,shape)
            self.input.append(buffer)

        for ov in loads:
            v = copy.copy(ov)
            if v in c_graph.egraph.produced_by:
                v = c_graph.egraph.produced_by[v][0].name
            tv = c_graph.constant_nodes[v]

            buffer = None
            if isinstance(tv, onnx.onnx_ml_pb2.NodeProto):
                if tv.attribute[0].type == 2:
                    data = np.array([tv.attribute[0].i])
                    buffer = ComputeBuffer(ov, data=data)
                elif tv.attribute[0].type == 4:
                    tv = tv.attribute[0].t
                elif tv.attribute[0].type == 1:
                    data = np.array([tv.attribute[0].f])
                    buffer = ComputeBuffer(ov, data=data)
                else:
                    raise Exception("not support")

            if not buffer:
                if isinstance(tv, onnx.onnx_ml_pb2.TensorProto):
                    dim = tv.dims
                    dtype = tv.data_type
                    buffer = ComputeBuffer(ov, data=onnx.numpy_helper.to_array(tv))
                else:
                    raise Exception("not support")
            cached_buffer[v] = buffer
            self.load[ov] = buffer

        self.insert_load_and_store(global_buffer, c_graph)
        pass


class InterGroupStrategy(object):
    def __init__(self):
        self.count = 0

    def get_unique_var_name(self, prefix):
        self.count += 1
        return prefix + str(self.count)

    def can_fusion(self, node1, node2):
        if node1.op_type in node_sets.ElementWiseNodeSet():
            return True
        return False

    def do_fusion(self, nodes):
        before_fusion_groups = deque()
        after_fusion_groups = deque()

        for node in nodes:
            before_fusion_groups.append([node])

        while len(before_fusion_groups) > 1:
            node1 = before_fusion_groups.popleft()
            node2 = before_fusion_groups.popleft()
            if self.can_fusion(node1[-1].current_node, node2[-1].current_node):
                node1.extend(node2)
                before_fusion_groups.appendleft(node1)
            else:
                after_fusion_groups.append(node1)
                before_fusion_groups.appendleft(node2)
        after_fusion_groups.extend(before_fusion_groups)

        fusion_blocks = []
        for group in after_fusion_groups:
            fusion_blocks.append(ExecutionBlock(group))
        return fusion_blocks
