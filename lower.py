import common
import numpy as np
from abc import ABCMeta, abstractmethod
from enum import Enum
from collections import defaultdict, deque,OrderedDict
from typing import Union, List, Tuple, Dict
import sympy_utils
import sympy
from sympy.codegen.rewriting import create_expand_pow_optimization
import copy
import re

import onnx
import onnx.numpy_helper

import node_sets


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
        self.count = 0
        self.type_and_shape = OrderedDict()

    def get_unique_var_name(self, prefix):
        self.count += 1
        return prefix + str(self.count)

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
            if node.op_type == "ReduceMean":
                imtermid = self.get_unique_var_name("sum_out")
                axes_out = self.get_unique_var_name("axes_in_reduceMean")
                axes_v = onnx.helper.make_tensor(
                    name="axes",
                    data_type=onnx.TensorProto.INT64,
                    dims=(1,),
                    vals=np.array([node.attribute[0].ints[0]]),
                )
                axes_node = onnx.helper.make_node(
                    "Constant",
                    [],
                    [axes_out],
                    f"axes_in_reduceMean_{node.name}",
                    value=axes_v,
                )
                sum_node = onnx.helper.make_node(
                    "ReduceSum",
                    [node.input[0], axes_out],
                    [imtermid],
                    f"sum/decomposed_from_{node.name}",
                )

                v = onnx.helper.make_tensor(
                    name="last_dim",
                    dims=(),
                    data_type=onnx.TensorProto.FLOAT,
                    vals=np.array([768.0]),
                )
                imtermid_1 = self.get_unique_var_name("constant_shape_in_axis")
                n_elem = onnx.helper.make_node(
                    "Constant",
                    [],
                    [imtermid_1],
                    name=f"shape[-1]/decomposed_from_{node.name}",
                    value=v,
                )
                div_node = onnx.helper.make_node(
                    "Div",
                    [imtermid, imtermid_1],
                    node.output,
                    f"Div/decomposed_from_{node.name}",
                )

                return [axes_node, sum_node, n_elem, div_node]
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
            symbol_shapes=[]
            for x in shape:
                if isinstance(x, str):
                    symbol_shape = sympy.Symbol(x)
                else:
                    symbol_shape= sympy.Integer(x)
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


class IRNode:
    def __init__(self):
        self.parent = None

    @abstractmethod
    def code_spice(self, var_map: dict = None, indent: int = 0):
        pass


class LoopAttr(Enum):
    Parallel = 1
    Reduce = 2
    Vectorization = 3


class Loop(IRNode):
    def __init__(self):
        self.var: sympy.Expr = None
        self.start = sympy.Integer(0)
        self.end = sympy.Integer(0)
        self.step = sympy.Integer(1)
        self.body = None
        self.depth = 0
        self.parallel:bool = False
        self.parallel_nest_loop:Loop = None
        self.attributes = LoopAttr.Parallel
        self.forward_var_set: Dict[ComputeBuffer] = OrderedDict()

    def code_spice(self, var_map: dict, indent: int = 0):
        dec_header = ""
        # forward declaration
        for fvar, buffer in self.forward_var_set.items():
            str_var = str(fvar)
            if buffer.shape is not None and buffer.shape[-1] == 1:
                dec_header += " " * indent + f"float {var_map[str_var]} = 0.0;\n"
            else:
                dec_header += (
                    " " * indent
                    + f"float e_{var_map[str_var]}[{buffer.shape[-1]}] = {{0.0}};\n"
                )
        src = dec_header
        src += " " * indent + f"//@{self.attributes.name}\n"
        src+= " " * indent
        if self.parallel:            
            p_var = f'{self.var}'
            p_var += f'_{self.parallel_nest_loop.var}' if self.parallel_nest_loop else ""
            src += f"for (int {p_var}={common.SpecialVar().parallel_loop_start}; {p_var}<{common.SpecialVar().parallel_loop_end}; {p_var}+={self.step}){{\n"
            if self.parallel_nest_loop:
                indents = " " * (indent*2)
                src += indents+f"auto {self.var} = {p_var}/{self.parallel_nest_loop.end};\n"
                nest_var =self.parallel_nest_loop.var
                src += indents+f"auto {nest_var} = {p_var}%{self.parallel_nest_loop.end};\n"
        else:
            src += f"for (int {self.var}={self.start}; {self.var}<{self.end}; {self.var}+={self.step}){{\n"
        if isinstance(self.body, list):
            for g in self.body:
                src += g.code_spice(var_map, indent + 4)
        else:
            src += self.body.code_spice(var_map, indent + 4)
        return src + " " * indent + "}\n"


class FunctionNode(IRNode):
    def __init__(self, inputs, outputs):
        self.input = inputs
        self.output = outputs
        self.name: str = ""
        self.const_var = []
        self.shape_var = []
        self.body: List[ExecutionBlock] = None

    def lower(self):
        assert len(self.body) == 1, "multiple body not supported in function node"
        shape_var = [i for i in self.body[0].shape if i.is_symbol]
        self.shape_var = list(set(shape_var))
        self.shape_var.sort(key=shape_var.index)

        self.body[0].gen_var(self.const_var)

    def code_spice(self, var_map: dict, indent: int = 0):
        if not var_map:
            var_map = self.body[0].var_map
        # self.output[0].type.tensor_type.shape.dim[1]

        # in_param = [f"const float* e_{var_map[i.name]}" for i in self.input]

        in_param = [f"const float** {common.SpecialVar().input_args}", f"int {common.SpecialVar().input_args_size}"]
        if self.body[0].body.parallel:
            in_param += [f'int64_t {common.SpecialVar().parallel_loop_start}, int64_t {common.SpecialVar().parallel_loop_end}']
        in_param += [f"const int64_t {i}" for i in self.shape_var]
        in_param = ",".join(in_param)
        
        
        #out_param = ",".join([f"float* e_{var_map[i.name]}" for i in self.output])
        out_param = f"float** output_args"

        func_signature = f"int {self.name}({in_param}, {out_param}) {{\n"

        code = ""
        code += func_signature
        indent += 4

        # DEBUG code
        assert_code = f"""
    #ifdef DEBUG_
    if ({common.SpecialVar().input_args_size} != {len(self.input)}){{
        printf(" assert {common.SpecialVar().input_args_size} != {len(self.input)} failed, please check your model or code\\n\\n");
        abort();
    }}
    #endif
"""
        code += assert_code

        parse_input = [
            " " * indent + f"const float* e_{var_map[i.name]} = {common.SpecialVar().input_args}[{idx}];"
            for idx, i in enumerate(self.input)
        ]
        code += "\n".join(parse_input) + "\n\n"

        parse_output = [
            " " * indent + f"float* e_{var_map[i.name]} = {common.SpecialVar().output_args}[{idx}];"
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
                x_arrstr = np.char.mod("%.6e", np_array)
                x_str = ",".join(x_arrstr)
                #################
                #x_str='0'
                #################
                if np_array.size == 1:
                    const_declare = (
                        f"static constexpr float e_{var_map[name]} = {x_str};\n"
                    )
                else:
                    const_declare = (
                        f"static constexpr float e_{var_map[name]}[] = {{{x_str}}};\n"
                    )
            elif isinstance(const, onnx.NodeProto):
                if const.attribute[0].type == 2:
                    v = const.attribute[0].i
                    const_declare = f"static constexpr float e_{var_map[name]} = {v};\n"
                elif const.attribute[0].type == 1:
                    v = const.attribute[0].f
                    const_declare = (
                        f"static constexpr float e_{var_map[name]} = {v}f;\n"
                    )
                else:
                    raise Exception("not supported")
            else:
                raise Exception("not supported")
            code += " " * indent + const_declare

        code += self.body[0].code_spice({}, indent)
        code += " " * indent + "return 12;\n"
        code += "}\n"

        return code


class ModuleNode(IRNode):
    def __init__(self):
        self.body: List[FunctionNode] = []

    def lower(self, function_recipes: list, func: callable):
        for function_recipe in function_recipes:
            function: FunctionNode = func(*function_recipe)
            self.body.append(function)

    def code_spice(self, var_map: dict, indent: int = 0):
        code = '#include <cmath>\nextern "C"{\n'

        for idx, func in enumerate(self.body):
            code += f"//the {idx}th function/sub_graph\n"
            code += func.code_spice(var_map, indent)
        # extern C
        code += "}\n"
        return code


class ComputeNode(IRNode):
    def __init__(self, op_type, inputs, outputs):
        self.op_type_ = op_type
        self.input = inputs
        self.output = outputs

    @property
    def op_type(self):
        return self.op_type_

    def gen_cpp_code_for_group(self, var_map: dict):
        named_vars_i = [var_map[i] for i in self.input]
        named_vars_o = [var_map[i] for i in self.output]
        assert len(named_vars_o) == 1
        assert len(named_vars_o) == 1
        named_var_o = named_vars_o[0]
        src = "auto "
        if self.op_type == "Add":
            src += f"{named_var_o} = {named_vars_i[0]} + {named_vars_i[1]};\n"
        elif self.op_type == "Sub":
            src += f"{named_var_o} = {named_vars_i[0]} - {named_vars_i[1]};\n"
        elif self.op_type == "Div":
            src += f"{named_var_o} = {named_vars_i[0]} / {named_vars_i[1]};\n"
        elif self.op_type == "Mul":
            src += f"{named_var_o} = {named_vars_i[0]} * {named_vars_i[1]};\n"
        elif self.op_type == "Pow":
            src += f"{named_var_o} = pow({named_vars_i[0]},{named_vars_i[1]});\n"
        elif self.op_type == "Sqrt":
            src += f"{named_var_o} = sqrt({named_vars_i[0]});\n"
        #elif self.op_type == "Cast":
        #    src += f"{named_var_o} = sqrt({named_vars_i[0]});\n"
        else:
            raise Exception("not supported")
        return src

    def code_spice(self, var_map: dict, indent: int = 0):
        return " " * indent + self.gen_cpp_code_for_group(var_map)


class ReduceNode(ComputeNode):
    def __init__(self, body: ComputeNode, axis=-1):
        self.axis = axis
        self.body: ComputeNode = body
        self.input = body.input
        self.output = body.output

    def code_spice(self, var_map: dict, indent: int = 0):
        code = "\n"
        # assert len(self.input) == 1
        # assert len(self.input) == 1 or (self.input[1] == -1)
        if self.body.op_type == "ReduceSum":
            named_var_i = var_map[self.input[0]]
            named_var_o = var_map[self.output[0]]
            if named_var_i != self.input[0]:
                code += " " * indent + f"// {named_var_i} = {self.input[0]};\n"
                code += " " * indent + f"// {named_var_o} = {self.output[0]};\n"
            code += " " * indent + f"{named_var_o} += {named_var_i};\n"
        else:
            raise Exception("not supported")
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

    def code_spice(self, named_var: str, buf: ComputeBuffer):
        if buf.data is not None and buf.data.size == 1:
            return f"{named_var}"
        else:
            shape = buf.shape or (buf.data is not None and buf.data.shape) or [1]
            index_of_dim_1 =[i for i in range(len(shape)) if shape[i]==1]
            stride = self.cal_stride(shape)
            index: sympy.Expr = buf.loop_index or [
                sympy_utils.sympy_symbol(f"i_{i}")
                for i in range(len(shape) - 1, -1, -1)
            ]
            if len(index) > len(shape):
                index = index[len(index) - len(shape) :]
            #broadcast handling
            br_index = [v for idx,v in enumerate(index) if idx not in index_of_dim_1]
            br_stride = [v for idx,v in enumerate(stride) if idx not in index_of_dim_1]

            res = sympy.Matrix([br_index]).dot(sympy.Matrix(br_stride))
            #res = res.subs(shape[0], 1)
            gs= re.findall('([a-zA-Z0-9_]+)\*\*(\d)', str(res))
            assert gs==[] #or gs[0][1] == '2', f"TODO fix me when pow {gs[0][1]} or other"
            #res= re.sub('([a-zA-Z0-9_]+)\*\*(\d)','\g<1>*\g<1>',str(res))            
            return f"{named_var}[{res}]"
        pass


class LoadNode(IRNode):
    def __init__(self, buf: ComputeBuffer):  # ComputeBuffer
        self.from_buf = buf
        self.to_buf = "to"

    @property
    def op_type(self):
        return "Load"

    def code_spice(self, var_map: dict, indent: int = 0):
        code = " " * indent
        var_name = self.from_buf.name
        assert var_name in var_map, f"name {var_name} not found in var_map"
        named_var = var_map[var_name]

        if named_var != var_name:
            code += f"//load ... {var_name} = {named_var};\n" + " " * indent
        annotated_var = Indexer().code_spice(named_var, self.from_buf)
        return code + f"auto {named_var} = e_{annotated_var};\n"


class StoreNode(IRNode):
    def __init__(self, buf: ComputeBuffer):  # ComputeBuffer
        self.to_var = buf

    @property
    def op_type(self):
        return "Store"

    def code_spice(self, var_map: dict, indent: int = 0):
        code = ""
        var_name = self.to_var.name
        assert var_name in var_map
        named_var = var_map[var_name]

        if named_var != var_name:
            code += " " * indent + f"// store ....{var_name} = {named_var};\n"
        annotated_var = Indexer().code_spice(named_var, self.to_var)
        return code + " " * indent + f"e_{annotated_var} = {named_var};\n"


class ExecutionBlock(IRNode):
    def __init__(self, group: List[IRNode]):
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
        self.forward_var_set = OrderedDict()
        self.body = None

        self.group = self.translate(group)

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
        loop.forward_var_set = self.forward_var_set

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
            if name[0].isdigit():
                name = "_" + name
            while name in exist_var:
                name = name + "_1"
            exist_var.add(name)
            return name

        for inp in self.forward_var_set:
            self.var_map[inp] = legal_name(inp)
        for inp in self.input:
            self.var_map[inp.name] = legal_name(inp.name)
        for out in self.output:
            self.var_map[out.name] = legal_name(out.name)
        for var in self.intermediate_var:
            self.var_map[var] = legal_name(var)
        for var in self.load:
            self.var_map[var] = legal_name(var)

        for out in external_var:
            self.var_map[out.name] = legal_name(out.name)

    def code_spice(self, var_map: dict, indent: int = 0):

        src = ""
        src += self.body.code_spice(self.var_map, indent)
        return src

    def lower(self):
        # assume shape [N, len, hidden]
        # add Loop()
        self.body = self.build_loop()

    def translate(self, group: List[Node]):
        new_group = []
        for g in group:
            node = g.current_node
            ir_node = ComputeNode(node.op_type, node.input, node.output)
            if node in node_sets.ReduceNodeSetInternal():
                self.has_reduce = True
                for o in node.output:
                    self.forward_var_set[o] = ComputeBuffer(
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
                    new_group.append(LoadNode(load_buf))
            new_group.append(g)
            for out in g.output:
                if out in output_name_map and not isinstance(g, ReduceNode):
                    new_group.append(StoreNode(output_name_map[out]))

        self.group = new_group

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
                self.intermediate_var[i]=0
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
                print("intermediate value appears in output, skip load", v)
                continue
            if v in cached_buffer:
                type_and_shape = c_graph.egraph.tensor_type_shape_info[v]
                assert v in cached_buffer, "buffer is not created"
                assert (
                    type_and_shape[1][-1] == cached_buffer[v].shape[-1]
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
