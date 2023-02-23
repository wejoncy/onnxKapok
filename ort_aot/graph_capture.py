import onnxruntime as ort
import onnx
from onnxsim import simplify

import tempfile
import numpy as np
from collections import deque, OrderedDict,defaultdict
from queue import PriorityQueue
from typing import Union, List, Tuple, Dict
from pathlib import Path
import re
import struct
import transformers

import common
import backend
import node_sets
from logger import logger


def remove_unused_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    out_to_node = {}
    nodename_to_index = {}
    for idx, n in enumerate(model.graph.node):
        for oname in n.output:
            assert oname not in out_to_node
            out_to_node[oname] = n
            nodename_to_index[n.name] = idx
    useful_node_names = []
    w = [out_to_node[o.name] for o in model.graph.output]
    while len(w):
        node = w.pop()
        useful_node_names.append(node.name)
        w.extend([out_to_node[i] for i in node.input if i in out_to_node])
    for i in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[i]
        if node.name not in useful_node_names:
            model.graph.node.pop(i)
            # print("node", i, "removed")
    return model


class CaptureOnnxSubGraph(object):
    def __init__(self, ort_optimize_first: bool = False):
        self.ort_optimize_first = ort_optimize_first
        self.graph = None
        self.model_proto = None
        self.node_order = dict()
        self.fused_node_nums = 0
        self.in_graph: common.OnnxInGraph = None

    def load(self, model_path):
        if self.ort_optimize_first:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
            with tempfile.NamedTemporaryFile(suffix='.onnx') as temp:
                session_options.optimized_model_filepath = temp.name
                f_model = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                    sess_options=session_options,
                )
                self.model_proto = onnx.shape_inference.infer_shapes(
                    onnx.load(temp.name), strict_mode=True
                )
        else:
            self.model_proto = onnx.shape_inference.infer_shapes(
                onnx.load(str(model_path)), strict_mode=True
            )
        self.graph = self.model_proto.graph
        self.in_graph = common.OnnxInGraph(self.model_proto)
        self.in_graph.gen_name2module_map()

        for node in self.graph.node:
            self.node_order[node.name] = len(self.node_order) + 1
        return self.graph

    def substitute_subgraph_with_fusednode(
        self, sub_graph: common.IndexSubGraph, lib_path: Path
    ) -> str:
        for node in sub_graph.sub_graph_nodes:
            self.graph.node.remove(self.in_graph.node_name2module[node.name])
            # for out in node.output:
            #    if out in self.produced_by:
            #        self.produced_by.pop(out)
            #    if out in self.consumed_by:
            #        self.consumed_by.pop(out)
            #    if out in self.node_name2module:
            #        self.node_name2module.pop(out)
            #    if out in self.initializer_name2module:
            #        self.initializer_name2module.pop(out)
        node_name = (
            sub_graph.sub_graph_nodes[0].name
            + "--->"
            + sub_graph.sub_graph_nodes[-1].name
        )
        func_name = re.sub(r"[^a-zA-Z0-9]", "_", node_name)
        
        def legal_shape(x):
            if isinstance(x, str):
                return re.sub(r'[^a-zA-Z0-9_]+', '_', x)
            return str(x)


        # we don't support when not all dynamic shapes existed in one input
        max_sym_len = 0
        max_sym_len_idx = 0
        for i0_idx,inp in enumerate(sub_graph.input_name_exclude_constant):
            if len([i for i in self.in_graph.tensor_type_shape_info[inp][1] if isinstance(i, str)]) > max_sym_len:
                max_sym_len= len([i for i in self.in_graph.tensor_type_shape_info[inp][1] if isinstance(i, str)])
                max_sym_len_idx = i0_idx

        in_symbol_name = OrderedDict()
        max_sym_inp = sub_graph.input_name_exclude_constant[max_sym_len_idx]
        for i1_idx, in_sp in enumerate(self.in_graph.tensor_type_shape_info[max_sym_inp][1]):
                if isinstance(in_sp, str):
                    if legal_shape(in_sp) not in in_symbol_name:
                        in_symbol_name[legal_shape(in_sp)] = [len(
                            in_symbol_name), (max_sym_len_idx, i1_idx)]

        dynamic_shape = []            
        for shape_symbol,pos in in_symbol_name.items():
            tb = struct.pack("<ii", *pos[1])
            tv = struct.unpack("<q", tb)
            dynamic_shape.append(tv[0])
                  
            
        output_shapes = []
        out_symbol_name = []

        for out0_idx,x in enumerate(sub_graph.output_name_ref_c):
            output_shapes.append([legal_shape(x1) for x1 in self.in_graph.tensor_type_shape_info[x][1]])
            for in_idx, x1 in enumerate(self.in_graph.tensor_type_shape_info[x][1]):
                if isinstance(x1, str):
                    out_sp = legal_shape(x1)
                    out_symbol_name.append((out0_idx, in_idx, in_symbol_name[out_sp][0]))
            output_shapes[-1] = ','.join(output_shapes[-1])
        
        packed_shape_match_pairs = []
        for s_m_p in out_symbol_name:
            tb = struct.pack("<HHHH", *s_m_p, 0)
            tv = struct.unpack("<q", tb)
            packed_shape_match_pairs.append(tv[0])
        
        node = onnx.helper.make_node(
            op_type="AOTanyOp",
            inputs=list(sub_graph.input_name_exclude_constant),
            outputs=list(sub_graph.output_name_ref_c.keys()),
            name=func_name,
            domain="com.microsoft",
            func_name=func_name,
            func_type=len(sub_graph.input_name_exclude_constant),
            lib_path=str(lib_path),
            output_shapes=output_shapes,
            match_pairs=packed_shape_match_pairs,
            dynamic_shape=dynamic_shape,
        )
        max_order = max([self.node_order[n.name] for n in sub_graph.sub_graph_nodes])
        self.node_order[node.name] = max_order
        self.graph.node.append(node)
        return func_name

    def create_model_wuth_sub_graph_(self, sub_graph: common.IndexSubGraph):
        sub_graph_nodes = sub_graph.sub_graph_nodes

        if len(sub_graph_nodes) <= 1 or len(sub_graph.input_name_exclude_constant) == 0:
            return None

        def get_input_name(sub_graph_nodes):
            input_name = set()
            tmp_output_name = set()
            for node in sub_graph_nodes:
                for inp in node.input:
                    input_name.add(inp)
                for out in node.output:
                    tmp_output_name.add(out)
            input_name = input_name - tmp_output_name
            return input_name

        input_name = get_input_name(sub_graph_nodes)
        assert set(input_name) == set(
            sub_graph.input_name_ref_c.keys()), "input name not match"
        assert all([x in input_name for x in sub_graph.input_name_exclude_constant]), "input name not match"

        type_shape_info = self.in_graph.tensor_type_shape_info
        onnx_sub_graph = onnx.GraphProto()
        onnx_sub_graph.name = (
            sub_graph_nodes[0].name + "--->" + sub_graph_nodes[-1].name
        )
        for inp in sub_graph.input_name_exclude_constant:
            dtype = type_shape_info[inp][0]
            tensor_type = onnx.helper.make_tensor_type_proto(
                elem_type=dtype, shape=type_shape_info[inp][1]
            )
            onnx_sub_graph.input.append(onnx.helper.make_value_info(inp, tensor_type))

        for out, _ in sub_graph.output_name_ref_c.items():
            dtype = type_shape_info[out][0] if out in type_shape_info else 0 # unkown dtype
            shape = type_shape_info[out][1] if out in type_shape_info else []
            if (shape == [] or shape == [1]) and dtype == 7:
                return None
            tensor_type = onnx.helper.make_tensor_type_proto(
                elem_type=dtype, shape=type_shape_info[out][1]
            )
            onnx_sub_graph.output.append(onnx.helper.make_value_info(out, tensor_type))

        onnx_sub_graph.node.extend(sub_graph_nodes)
        for name in sub_graph.input_name_ref_c.keys():
            if name not in sub_graph.input_name_exclude_constant:
                if name in self.in_graph.produced_by:
                    onnx_sub_graph.node.extend(self.in_graph.produced_by[name])
                elif 'out_'+name in self.in_graph.graph_input_names:
                    pass
                else:
                    onnx_sub_graph.initializer.append(
                        self.in_graph.initializer_name2module[name]
                    )
        onnx_sub_graph.node.sort(key=lambda x: self.node_order[x.name])
        opset_imports = [
            onnx.helper.make_operatorsetid(domain, opset)
            for domain, opset in {"": 17, "com.microsoft": 1}.items()
        ]
        subgraph_model = onnx.helper.make_model(
            onnx_sub_graph, opset_imports=opset_imports
        )
        subgraph_model = onnx.shape_inference.infer_shapes(
            subgraph_model, strict_mode=True
        )
        onnx.checker.check_model(subgraph_model)
        # onnx.save(subgraph_model, "subgraph.onnx")

        # self.substitute_subgraph_with_fusednode(sub_graph)
        # onnx.checker.check_model(self.model_proto)
        # save_path = './fused.onnx'
        # onnx.save(self.model_proto, save_path)
        return subgraph_model

    def is_constant_input(self, tensor_name: str):
        if tensor_name in self.in_graph.initializer_name2module:
            return True
        if (
            tensor_name in self.in_graph.produced_by
            and self.in_graph.produced_by[tensor_name][0].op_type == "Constant"
        ):
            return True
        # if tensor_name not in self.produced_by:
        #    return True
        return False

    def verify_input_is_not_in_cycle(
        self, node, future_inputs, all_inputs_in_sub_graph: set, depth=13
    ):
        def verify_internal(in_node: str, i_depth):
            if i_depth == 0:
                return False

            for inp in in_node.input:
                if self.is_constant_input(inp) or inp in all_inputs_in_sub_graph:
                    continue
                if (
                    inp in future_inputs
                    or verify_internal(self.in_graph.produced_by[inp][0], i_depth - 1)
                    is False
                ):
                    return False
            return True

        for inp in node.input:
            if (
                inp in all_inputs_in_sub_graph
                or inp in future_inputs
                or self.is_constant_input(inp)
                or verify_internal(self.in_graph.produced_by[inp][0], depth)
            ):
                continue
            else:
                return False

        return True

    def run(self, model_path: Path, lib_path: Path):
        self.load(model_path)
        e_node_set = node_sets.ElementWiseNodeSet().type_collection.union(
            node_sets.ReduceNodeSet(self.in_graph.produced_by).type_collection
        )
        assigned_node_by_name = set()
        available_tensor = set(i.replace('out_','') for i in self.in_graph.graph_input_names)
        not_available_tensor = set()

        def find_sub_graph_by_dfs(
            q_nodes: PriorityQueue,
            sub_graph: common.IndexSubGraph,
        ):
            _, node = q_nodes.get()
            if node.name in assigned_node_by_name:
                return None, None

            if (
                self.verify_input_is_not_in_cycle(
                    node, not_available_tensor, available_tensor
                )
                is False
            ):
                return

            assigned_node_by_name.add(node.name)

            for i in node.input:
                not_available_tensor.add(i)

            for o in node.output:
                not_available_tensor.add(o)

            if node.op_type == 'Cast' and (
                self.in_graph.tensor_type_shape_info[node.input[0]][0] == 7):
                return None, None
                
            sub_graph.sub_graph_nodes.append(node)

            for name in node.input:
                if (
                    name in self.in_graph.produced_by
                    and name not in assigned_node_by_name
                    and not self.is_constant_input(name)
                ):
                    pre_nodes = self.in_graph.produced_by[name]
                    if (
                        pre_nodes[0].op_type != "Constant"
                        and pre_nodes[0].op_type in e_node_set
                    ):
                        q_nodes.put((self.node_order[pre_nodes[0].name], pre_nodes[0]))

                if name not in sub_graph.input_name_ref_c:
                    sub_graph.input_name_ref_c[name] = 0
                sub_graph.input_name_ref_c[name] += 1

            connected_node_o = []
            for name in node.output:
                if (
                    name in self.in_graph.consumed_by
                    and name not in assigned_node_by_name
                ):
                    next_nodes = self.in_graph.consumed_by[name]
                    connected_node_o += next_nodes
                    for con_node in next_nodes:
                        if con_node.op_type in e_node_set:
                            q_nodes.put((self.node_order[con_node.name], con_node))

                if name not in sub_graph.output_name_ref_c:
                    sub_graph.output_name_ref_c[name] = len(self.in_graph.consumed_by[name])


            while not q_nodes.empty():
                find_sub_graph_by_dfs(q_nodes, sub_graph)


        node_nums_before_fusion = len(self.graph.node)
        sub_graph_list = []
        sub_model_list = []
        for node in self.graph.node:
            for i in node.input:
                available_tensor.add(i)

            if node.name in assigned_node_by_name:
                continue
            elif node.op_type in e_node_set:
                sub_graph = common.IndexSubGraph()
                q_nodes = PriorityQueue()

                q_nodes.put((self.node_order[node.name], node))
                find_sub_graph_by_dfs(q_nodes, sub_graph)
                available_tensor.update(not_available_tensor)
                not_available_tensor = set()
                sub_graph.sub_graph_nodes.sort(key=lambda x: self.node_order[x.name])
                sub_graph.analyze_input_output(self.in_graph.tensor_type_shape_info, self.is_constant_input)

                sub_model = self.create_model_wuth_sub_graph_(sub_graph)
                if sub_model:
                    sub_graph_list.append(sub_graph)
                    sub_model_list.append(sub_model)
                    if len(sub_model_list) == 1000:
                        break
            else:  # not element-wise node
                assigned_node_by_name.add(node.name)
                for o in node.output:
                    available_tensor.add(o)

        model_with_name = OrderedDict()
        for idx, (sub_graph, sub_model) in enumerate(
            zip(sub_graph_list, sub_model_list)
        ):
            #if idx != 1:continue
            func_name = self.substitute_subgraph_with_fusednode(sub_graph, lib_path)
            model_with_name[func_name] = sub_model

        self.graph.node.sort(key=lambda x: self.node_order[x.name])
        node_nums_after_fusion = len(self.graph.node)
        self.fused_node_nums = node_nums_before_fusion - node_nums_after_fusion
        logger.info(
            f"after fusion, count of nodes reduces from {node_nums_before_fusion} to {node_nums_after_fusion} "
        )
        return model_with_name
