from . import common
from .logger import logger
from .ir import ExecutionBlock, ComputeBuffer, ReduceNode, ComputeNode
from . import de_compose
from . import utils
from . import node_sets

import onnx
from typing import Union, List, Tuple, Dict, Set
from collections import defaultdict, deque, OrderedDict
import copy
import types


class Node(object):
    def __init__(self, node, input_nodes=None, input_constant=None, output_nodes=None):
        self.current_node = node
        self.input_nodes = input_nodes if input_nodes else []
        self.input_constant = input_constant if input_constant else []
        self.output_nodes = output_nodes if output_nodes else []
        self.output_with_shapes = OrderedDict()
        self.input_with_shapes = OrderedDict()

    @property
    def op_type(self):
        if isinstance(self.current_node, onnx.onnx_ml_pb2.ValueInfoProto):
            return "PlaceHolder"
        elif isinstance(self.current_node, onnx.onnx_ml_pb2.TensorProto):
            return "Tensor"
        return self.current_node.op_type

    @property
    def name(self):
        if self.op_type == "PlaceHolder":
            return 'out_' + self.current_node.name
        return self.current_node.name

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        elif isinstance(other, str):
            return self.name == other
        try:
            assert self.current_node.HasField("op_type")
        except BaseException:
            return False

        return self.current_node.op_type == other.op_type and self.current_node.name == other.name

    def __hash__(self):
        return hash(self.name + self.op_type)


class ConnectionGraph(object):
    def __init__(self, model):
        self.model = model
        self.egraph: common.OnnxInGraph = None
        self.node_collection = []
        self.node_2_gnode = {}
        self.in_dgree = defaultdict(lambda: 0)
        self.entry_nodes = []
        self.constant_nodes = OrderedDict()
        self.type_and_shape = OrderedDict()
        self.decompose_dispatcher = de_compose.DecomposeDispatch()

    # God, ORT will produce a weird case: node's name is same as node's output name
    def get_or_create_gnode(self, node, prefix=""):
        name = prefix + node.name
        if name not in self.node_2_gnode:
            self.node_2_gnode[name] = Node(node)
            if isinstance(node, onnx.NodeProto):
                self.node_2_gnode[name].input_with_shapes = {
                    inp: self.type_and_shape[inp] for inp in node.input
                    # is graph_input or is not constant
                    if inp in self.type_and_shape and inp not in self.egraph.initializer_name2module and
                    (inp not in self.egraph.produced_by or self.egraph.produced_by[inp][0].op_type != "Constant")
                }
                self.node_2_gnode[name].output_with_shapes = {inp: self.type_and_shape[inp] for inp in node.output}

        return self.node_2_gnode[name]

    def try_decompose(self, node, shape_info_map) -> (list):
        if node in node_sets.DecomposeNodeSetInternal():
            return self.decompose_dispatcher(node, shape_info_map=shape_info_map)

        return []

    def decompose(self, e_graph: common.OnnxInGraph, recursive_depth=1):
        replace_nodes = {}
        graph = e_graph.graph
        modified = False
        for node in graph.node:
            r_nodes = self.try_decompose(node, e_graph.tensor_type_shape_info)
            if r_nodes:
                replace_nodes[node.name] = (node, r_nodes)
                modified = True
        for node, r_nodes in replace_nodes.values():
            graph.node.remove(node)
            graph.node.extend(r_nodes)
        if recursive_depth > 0 and modified:
            self.decompose(e_graph, recursive_depth - 1)

    def try_recompose(self, node: onnx.NodeProto, egraph: common.OnnxInGraph) -> (list):
        assert node.op_type == "Erf"
        if len(node.input) != 1:
            return []
        if len(node.output) != 1:
            return []
        input_div_node = egraph.produced_by[node.input[0]][0]
        if input_div_node.op_type != "Div":
            return []
        if len(egraph.consumed_by[node.output[0]]) != 1:
            return []
        output_add_node = egraph.consumed_by[node.output[0]][0]
        if output_add_node.op_type != "Add":
            return []
        output_mul_node = egraph.consumed_by[output_add_node.output[0]][0]
        if output_mul_node.op_type != "Mul" or (input_div_node.input[0] not in output_mul_node.input):
            return []
        mul_0_5_node = egraph.consumed_by[output_mul_node.output[0]][0]
        if mul_0_5_node.op_type != "Mul":
            return []
        gelu_node = onnx.helper.make_node("Gelu", [input_div_node.input[0]], [mul_0_5_node.output[0]])
        return ([input_div_node, node, output_add_node, output_mul_node, mul_0_5_node], gelu_node)

    # TODO: implement recompose, Gelu, the reason is Gelu has a Fast algorithm.
    def recompose(self, egraph: common.OnnxInGraph):
        replace_nodes = {}
        graph: onnx.GraphProto = egraph.graph
        erf_bool = ['Erf' in node.op_type for node in graph.node]
        erf_index = erf_bool.index(True) if True in erf_bool else -1
        if erf_index >= 0:
            assert sum(erf_bool) == 1, "not support multiple erf node"
            node = graph.node[erf_index]
            r_nodes, new_node = self.try_recompose(node, egraph)
            if r_nodes:
                replace_nodes[node.name] = (r_nodes, new_node)
        for r_nodes, node in replace_nodes.values():
            for t_b_n in r_nodes:
                graph.node.remove(t_b_n)
            graph.node.append(node)

    def build_relationship(self):
        self.egraph = common.OnnxInGraph(self.model)
        self.egraph.gen_name2module_map()
        self.decompose(self.egraph)
        self.recompose(self.egraph)
        self.egraph.gen_name2module_map()
        egraph = self.egraph
        self.type_and_shape = egraph.tensor_type_shape_info
        self.constant_nodes.update(egraph.initializer_name2module)
        for node in egraph.graph.node:
            if node.op_type == "Constant":
                self.constant_nodes[node.name] = node

        for node in egraph.graph.node:
            if node.op_type == "Constant":
                continue
            gnode: Node = self.get_or_create_gnode(node)
            self.node_collection.append(gnode)
            for inp in node.input:
                if inp in egraph.produced_by:
                    in_gnode = self.get_or_create_gnode(
                        egraph.produced_by[inp][0])
                    if in_gnode.name in self.constant_nodes:
                        gnode.input_constant.append(in_gnode)
                    else:
                        gnode.input_nodes.append(in_gnode)
                elif inp in egraph.initializer_name2module:
                    in_gnode = self.get_or_create_gnode(
                        egraph.initializer_name2module[inp]
                    )
                    gnode.input_constant.append(in_gnode)
                elif "out_" + inp in egraph.graph_input_names:
                    in_gnode = self.get_or_create_gnode(
                        egraph.node_name2module["out_" + inp])
                    in_gnode.output_nodes.append(gnode)
                    self.in_dgree[gnode] += 1
                    gnode.input_nodes.append(in_gnode)
                    if in_gnode in self.entry_nodes:
                        continue
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
                        egraph.node_name2module["out_" + out], prefix="out_"
                    )
                    out_gnode.input_nodes.append(gnode)
                    if gnode != out_gnode:
                        gnode.output_nodes.append(out_gnode)
                        self.in_dgree[out_gnode] += 1
                    else:
                        print(' ')
                else:
                    raise Exception("output not found")
        # for inp in egraph.graph_input_names:
        #    in_gnode:Node = self.get_or_create_gnode(egraph.node_name2module[inp])
        #    len(in_gnode.output_nodes) == 2
        #    [i.name for i in in_gnode.output_nodes]


################################################################################
# Node define in execution_planer.py
def translate_in_out_to_ComputeBuffer(node: Node, buffer_cache: dict) -> (List[ComputeBuffer], List[ComputeBuffer]):
    input_buffer = []

    for inp in node.current_node.input:
        deco_name = inp if inp in buffer_cache else "out_" + inp
        if deco_name in buffer_cache:
            input_buffer.append(buffer_cache[deco_name])
        elif inp in node.input_with_shapes:
            input_buffer.append(ComputeBuffer(
                                name=inp,
                                dtype=common.TENSOR_TYPE_TO_NP_TYPE[node.input_with_shapes[inp][0]],
                                shape=node.input_with_shapes[inp][1]))
            buffer_cache[inp] = input_buffer[-1]
        else:
            tv = (node.input_constant[node.input_constant.index(inp)].current_node if inp in node.input_constant
                  else next(filter(lambda x: x.current_node.output[0] == inp, node.input_constant)).current_node)
            data = common.parse_onnx_to_numpyarray(tv)
            buffer = ComputeBuffer(inp, data=data)
            input_buffer.append(buffer)
            buffer_cache[inp] = input_buffer[-1]
    output_buffer = []
    for out in node.current_node.output:
        assert out in node.output_with_shapes
        deco_name = out if out in buffer_cache else "out_" + out
        if deco_name in buffer_cache:
            output_buffer.append(buffer_cache[deco_name])
        else:
            output_buffer.append(ComputeBuffer(
                name=out,
                dtype=common.TENSOR_TYPE_TO_NP_TYPE[node.output_with_shapes[out][0]],
                shape=node.output_with_shapes[out][1]))
            buffer_cache[out] = output_buffer[-1]

    return input_buffer, output_buffer


def lower_Node_to_IRNode(block: ExecutionBlock, graph_io: common.GraphIOBuffer, buffer_cache: Dict[str, ComputeBuffer]):
    group = block.group
    new_group = []
    for g in group:
        in_b, out_b = translate_in_out_to_ComputeBuffer(g, buffer_cache)
        ir_node = ComputeNode(g.op_type, in_b, out_b, g.name)
        for ib in in_b:
            ib.successor.append(ir_node)
        for ob in out_b:
            ob.predecessor = ir_node
        node = g.current_node
        if node in node_sets.ReduceNodeSetInternal():
            block.has_reduce = True
            for o in node.output:
                o_buf = out_b[out_b.index(o)]
                block.forward_var_set[0][o] = out_b
                # ComputeBuffer(name=o, dtype=g.output_with_shapes[o][0], shape=g.output_with_shapes[o][1])
            new_group.append(ReduceNode(ir_node))
        else:
            new_group.append(ir_node)
    block.fused_groups.append(new_group)
    block.group = new_group
################################################################################


class InterGroupStrategy(object):
    def __init__(self, target: str):
        self.count = 0
        self.target = target

    def can_fusion(self, node1, node2):
        if node1.op_type in node_sets.ElementWiseNodeSet():
            return True
        return False

    def do_fusion_for_triton(self, nodes):
        return [nodes]

    def do_fusion(self, nodes):
        if self.target == "triton":
            return self.do_fusion_for_triton(nodes)
        before_fusion_groups = deque()
        after_fusion_groups = deque()

        for node in nodes:
            before_fusion_groups.append([node])

        while len(before_fusion_groups) > 1:
            node1 = before_fusion_groups.popleft()
            node2 = before_fusion_groups.popleft()
            if self.can_fusion(node1[-1], node2[-1]):
                node1.extend(node2)
                before_fusion_groups.appendleft(node1)
            else:
                after_fusion_groups.append(node1)
                before_fusion_groups.appendleft(node2)
        after_fusion_groups.extend(before_fusion_groups)

        return after_fusion_groups


class ExecutionPrepare(object):
    def __init__(self, model, target: str):
        self.edge_graph: ConnectionGraph = ConnectionGraph(model)
        self.external_buffer: common.GraphIOBuffer = common.GraphIOBuffer()
        self.graph_io_name = set()
        self.target = target

    def prepare(self):
        self.edge_graph.build_relationship()

    def topological_with_reduce_last(self):
        queue = deque()
        for i in self.edge_graph.entry_nodes:
            queue.append(i)
        in_degree = copy.copy(self.edge_graph.in_dgree)
        # for i in self.edge_graph.node_collection:
        #    if self.edge_graph.in_dgree[i] == 0:
        #        queue.append(i)
        # if not queue:
        #    raise Exception("no node with in_degree 0")

        reduce_nodes = node_sets.ReduceNodeSet(
            self.edge_graph.egraph.produced_by)
        sorted_nodes = []

        def has_non_reduce_child(queue):
            for n in queue:
                if n.current_node not in reduce_nodes:
                    return True
            return False

        while queue:
            node: Node = queue.popleft()
            if node.current_node in reduce_nodes and has_non_reduce_child(queue):
                queue.append(node)
                continue
            # print(node.current_node.name)
            if node.name not in self.graph_io_name:
                sorted_nodes.append(node)
            for n in node.output_nodes:
                in_degree[n] -= 1
                if in_degree[n] == 0:
                    queue.append(n)
        assert all([v == 0 for v in in_degree.values()]), "Topological sort failed, as graph has cycle"
        return sorted_nodes

    def analyze_io_buffer(self, groups, analyze_io: callable):
        cached_buffer: Dict[str, ComputeBuffer] = OrderedDict()
        type_and_shape = self.edge_graph.type_and_shape
        for i in self.edge_graph.model.graph.input:
            ins = utils.convert_onnx_value_to_computebuffer(i, prefix='out_')
            self.external_buffer.var_buffer_in.append(ins)
            cached_buffer[ins.name] = ins
        for i in self.edge_graph.model.graph.output:
            outs = utils.convert_onnx_value_to_computebuffer(i, prefix='out_')
            self.external_buffer.var_buffer_out.append(outs)
            cached_buffer[outs.name] = outs
        for i in self.edge_graph.model.graph.initializer:
            self.external_buffer.const_buffer.append(i)
        for i in self.edge_graph.constant_nodes.values():
            self.external_buffer.const_buffer.append(i)

        for block in groups:
            lower_Node_to_IRNode(block, self.external_buffer, cached_buffer)

        # check all buffer without predecessor is placeholder
        for b in cached_buffer.values():
            if b.predecessor is None:
                assert b.name in self.edge_graph.egraph.graph_input_names or b.data is not None
            elif len(b.successor) == 0:
                assert b.name in self.edge_graph.egraph.graph_output_names

        for g in groups:
            analyze_io(g, self.external_buffer, self.edge_graph, cached_buffer, self.target)

    def create_execution_plan(self, analyze_io: callable):
        for i in self.edge_graph.model.graph.input:
            self.graph_io_name.add('out_' + i.name)
        for i in self.edge_graph.model.graph.output:
            self.graph_io_name.add('out_' + i.name)

        sorted_nodes = self.topological_with_reduce_last()

        intergroup_st = InterGroupStrategy(self.target)
        node_group = intergroup_st.do_fusion(nodes=sorted_nodes)

        # convert to IRNode/ ExecutionBlock
        fusion_blocks = []
        for group in node_group:
            fusion_blocks.append(ExecutionBlock(group))
        self.analyze_io_buffer(fusion_blocks, analyze_io)

        return fusion_blocks
