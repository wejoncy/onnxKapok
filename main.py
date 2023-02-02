import onnxruntime as ort
import onnx
from onnxsim import simplify

import time
import numpy as np
from collections import deque, OrderedDict
from queue import PriorityQueue
from typing import Union, List, Tuple, Dict
from pathlib import Path
import re

import transformers

import lower
import common
import backend
import node_sets


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
    def __init__(self):
        self.graph = None
        self.model_proto = None
        self.node_order = dict()

        self.in_graph: common.OnnxInGraph = None

    def load(self, model_path):
        self.model_proto = onnx.shape_inference.infer_shapes(
            onnx.load(model_path), strict_mode=True
        )
        self.graph = self.model_proto.graph
        self.in_graph = common.OnnxInGraph(self.model_proto)
        self.in_graph.gen_name2module_map()

        for node in self.graph.node:
            self.node_order[node.name] = len(self.node_order) + 1
        print("node count:", len(self.graph.node))
        return self.graph

    def substitute_nodes_with_subgraph(
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
        output_shapes = []
        for x in sub_graph.output_name_ref_c:
            output_shapes.append(
                ",".join([str(x1) for x1 in self.in_graph.tensor_type_shape_info[x][1]])
            )
        node = onnx.helper.make_node(
            op_type="AOTanyOp",
            inputs=list(sub_graph.input_name_exclude_constant.keys()),
            outputs=list(sub_graph.output_name_ref_c.keys()),
            name=node_name,
            domain="com.microsoft",
            func_name=func_name,
            func_type=len(sub_graph.input_name_exclude_constant.keys()),
            lib_path=str(lib_path),
            output_shapes=output_shapes,
        )
        max_order = max([self.node_order[n.name] for n in sub_graph.sub_graph_nodes])
        self.node_order[node.name] = max_order
        self.graph.node.append(node)
        return func_name

    def create_sub_graph_to_onnx(self, sub_graph: common.IndexSubGraph):
        sub_graph_nodes = sub_graph.sub_graph_nodes

        if len(sub_graph_nodes) == 1:
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
            sub_graph.input_name_ref_c.keys()
        ), "input name not match"
        # print(
        #    "graph inputs:",
        #    len(input_name),
        #    " var inputs",
        #    len(sub_graph.input_name_exclude_constant),
        #    "nodes:",
        #    len(sub_graph_nodes),
        # )

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
            dtype = type_shape_info[out][0]
            tensor_type = onnx.helper.make_tensor_type_proto(
                elem_type=dtype, shape=type_shape_info[out][1]
            )
            onnx_sub_graph.output.append(onnx.helper.make_value_info(out, tensor_type))

        onnx_sub_graph.node.extend(sub_graph_nodes)
        for name in sub_graph.input_name_ref_c.keys():
            if name not in sub_graph.input_name_exclude_constant:
                if name in self.in_graph.produced_by:
                    onnx_sub_graph.node.extend(self.in_graph.produced_by[name])
                else:
                    onnx_sub_graph.initializer.append(
                        self.in_graph.initializer_name2module[name]
                    )
        onnx_sub_graph.node.sort(key=lambda x: self.node_order[x.name])
        opset_imports = [
            onnx.helper.make_operatorsetid(domain, opset)
            for domain, opset in {"": 16, "com.microsoft": 1}.items()
        ]
        subgraph_model = onnx.helper.make_model(
            onnx_sub_graph, opset_imports=opset_imports
        )
        subgraph_model = onnx.shape_inference.infer_shapes(
            subgraph_model, strict_mode=True
        )
        onnx.checker.check_model(subgraph_model)
        # onnx.save(subgraph_model, "subgraph.onnx")

        # self.substitute_nodes_with_subgraph(sub_graph)
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

    def verify_input_is_not_in_cycle(self, node, all_inputs_in_sub_graph: set, depth=3):
        def verify_internal(in_node: str, i_depth):
            if i_depth == 0:
                return False

            for inp in in_node.input:
                if self.is_constant_input(inp):
                    continue
                if (
                    inp in all_inputs_in_sub_graph
                    or verify_internal(self.in_graph.produced_by[inp][0], i_depth - 1)
                    is False
                ):
                    return False
            return True

        for inp in node.input:
            if (
                inp in all_inputs_in_sub_graph
                or self.is_constant_input(inp)
                or verify_internal(self.in_graph.produced_by[inp][0], depth)
            ):
                continue
            else:
                return False

        return True

    def run(self):
        e_node_set = node_sets.ElementWiseNodeSet().type_collection.union(
            node_sets.ReduceNodeSet(self.in_graph.produced_by).type_collection
        )
        assigned_node_by_name = set()
        available_tensor = set()
        lib_path = Path(__file__).resolve(strict=True).parent / "libcode.so"

        def find_sub_graph_by_dfs(
            q_nodes: PriorityQueue,
            sub_graph: common.IndexSubGraph,
        ):
            _, node = q_nodes.get()

            if node.name in assigned_node_by_name:
                return None, None

            # if not sub_graph.input_name_exclude_constant:
            #    for name in (list(node.input) +self.graph_input_names):
            #        if self.is_constant_input(name) is False:
            #            sub_graph.input_name_exclude_constant.add(name)
            #        sub_graph.input_name_ref_c[name] += 0

            if self.verify_input_is_not_in_cycle(node, available_tensor) is False:
                return

            assigned_node_by_name.add(node.name)
            sub_graph.sub_graph_nodes.append(node)

            for i in node.input:
                available_tensor.add(i)

            for o in node.output:
                available_tensor.add(o)

            new_input_name = []
            connected_node_i = []
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
                        connected_node_i += pre_nodes
                        q_nodes.put((self.node_order[pre_nodes[0].name], pre_nodes[0]))
                if name not in sub_graph.input_name_ref_c:
                    sub_graph.input_name_ref_c[name] = 0
                sub_graph.input_name_ref_c[name] += 1

                if (
                    name in self.in_graph.produced_by
                    and name not in sub_graph.input_name_exclude_constant
                    and name not in sub_graph.output_name_ref_c
                    and not self.is_constant_input(name)
                ):
                    pre_nodes = self.in_graph.produced_by[name]
                    if pre_nodes[0].op_type != "Constant":
                        new_input_name.append(name)

            for v in new_input_name:
                sub_graph.input_name_exclude_constant[v] = 0

            if len(sub_graph.input_name_exclude_constant) > 4:
                return

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
                    sub_graph.output_name_ref_c[name] = 0
                sub_graph.output_name_ref_c[name] += 1

            while not q_nodes.empty():
                find_sub_graph_by_dfs(q_nodes, sub_graph)
            # for con_node in connected_node_i + connected_node_o:
            #    if con_node.op_type in e_node_set:
            #        find_sub_graph_by_dfs(con_node, sub_graph)
            #    else:
            #        assigned_node_by_name.add(node.name)

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
                sub_graph.sub_graph_nodes.sort(key=lambda x: self.node_order[x.name])
                sub_graph.analyze_input_output(self.in_graph.consumed_by)

                sub_model = self.create_sub_graph_to_onnx(sub_graph)
                if sub_model:
                    sub_graph_list.append(sub_graph)
                    sub_model_list.append(sub_model)
                    if len(sub_model_list) == 5000:
                        break
            else:  # not element-wise node
                assigned_node_by_name.add(node.name)
                for o in node.output:
                    available_tensor.add(o)

        model_with_name = OrderedDict()
        for idx, (sub_graph, sub_model) in enumerate(
            zip(sub_graph_list, sub_model_list)
        ):
            # if idx != 4:continue
            func_name = self.substitute_nodes_with_subgraph(sub_graph, lib_path)
            model_with_name[func_name] = sub_model
        target = "x86_64"
        #target = "aarch64"
        backend.CppBackend().compile(model_with_name, lib_path, target)

        test_lib(model_with_name, lib_path)

        self.graph.node.sort(key=lambda x: self.node_order[x.name])
        print("node count:", len(self.graph.node))

        # onnx.checker.check_model(self.model_proto)
        save_path = Path("./fused.onnx").resolve()
        opset = self.model_proto.opset_import.add()
        opset.version = 1
        opset.domain = "com.microsoft"
        model_simp = self.model_proto
        # model_simp, check = simplify(self.model_proto)
        # assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path)
        print(f"successful compiled to onnx model:{save_path} lib_path:{lib_path}")


class CostTime(object):
    def __init__(self, total_time):
        self.total_time = total_time
        self.st = 0

    def __enter__(self):
        self.st = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total_time[0] = time.perf_counter() - self.st


def test_lib(onnx_model_map: dict, lib_path: Path):
    if len(onnx_model_map) > 1:
        print(" not support test multi subgraph")
        return

    func_name, onnx_model = onnx_model_map.popitem()
    input_shapes = [
        [i.dim_param or i.dim_value for i in inp.type.tensor_type.shape.dim]
        for inp in onnx_model.graph.input
    ]
    output_shapes = [
        [i.dim_param or i.dim_value for i in out.type.tensor_type.shape.dim]
        for out in onnx_model.graph.output
    ]

    in_dynamic_shape_axis = [
        [idx for idx, i in enumerate(in_shape) if isinstance(i, str)]
        for in_shape in input_shapes
    ]
    out_dynamic_shape_axis = [
        [idx for idx, i in enumerate(out_shape) if isinstance(i, str)]
        for out_shape in output_shapes
    ]
    for input_shape, in_dy_axis in zip(input_shapes, in_dynamic_shape_axis):
        for dy in in_dy_axis:
            input_shape[dy] = 24
        if 0 in in_dy_axis:
            input_shape[0] = 1

    for output_shape, out_dy_axis in zip(output_shapes, out_dynamic_shape_axis):
        for dy in out_dy_axis[1:]:
            output_shape[dy] = 24
        if 0 in out_dy_axis:
            output_shape[0] = 1
    input_shapes = [tuple(input_shape) for input_shape in input_shapes]

    import ctypes

    so = ctypes.CDLL(str(lib_path))
    func_handle = getattr(so, func_name)

    # func_handle.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    func_handle.restype = ctypes.c_int
    a_in = []
    for input_shape in input_shapes:
        a_in.append(np.random.rand(*input_shape).astype(np.float32))

    c_out = []
    for output_shape in output_shapes:
        c_out.append(np.zeros(shape=output_shape).astype(np.float32))

    input_count = len(onnx_model.graph.input)
    output_count = len(onnx_model.graph.output)
    in_arg_type = ctypes.c_void_p * input_count
    out_arg_type = ctypes.c_void_p * output_count

    if input_count == 2:
        in_arg = in_arg_type(a_in[0].ctypes.data, a_in[1].ctypes.data)
    elif input_count == 3:
        in_arg = in_arg_type(
            a_in[0].ctypes.data, a_in[1].ctypes.data, a_in[2].ctypes.data
        )
    else:
        in_arg = in_arg_type(a_in[0].ctypes.data)

    if output_count == 2:
        out_arg = out_arg_type(c_out[0].ctypes.data, c_out[1].ctypes.data)
    else:
        out_arg = out_arg_type(c_out[0].ctypes.data)

    idx = [
        i
        for i in range(len(in_dynamic_shape_axis))
        if len(in_dynamic_shape_axis[i]) > 1
    ][0]
    in_dy_axis = in_dynamic_shape_axis[idx]
    input_shape = input_shapes[idx]
    c_tc = [0]
    with CostTime(c_tc) as tc:
        for i in range(10000):
            ret = func_handle(
                in_arg,
                input_count,
                0,
                input_shape[0] * input_shape[1],
                input_shape[in_dy_axis[0]],
                input_shape[in_dy_axis[1]],
                out_arg,
            )

    for idx, a_x in enumerate(a_in):
        a_x.tofile(f"a{idx}.bin")

    session = ort.InferenceSession(onnx_model.SerializeToString())
    input_feed = {i.name: a_in[idx] for idx, i in enumerate(session.get_inputs())}

    py_tc = [0]
    with CostTime(py_tc) as tc:
        for i in range(10000):
            out = session.run([i.name for i in session.get_outputs()], input_feed)

    all_passed = all(
        [np.allclose(out[i], c_out[i], rtol=1e-03, atol=1e-05) for i in range(len(out))]
    )
    if all_passed:
        print(f"Results are matched, time_cost: py_tc:{py_tc}, c_tc:{c_tc}")
    else:
        print("Results are not matched")


def topological_by_level():
    edge = [
        [0, 3],
        [1, 3],
        [2, 3],
        [3, 4],
        [3, 5],
        [4, 6],
        [6, 7],
        [7, 8],
        [5, 8],
        [8, 9],
    ]
    edge = [
        [0, 1],
        [0, 10],
        [10, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [2, 6],
        [3, 8],
        [4, 7],
        [5, 7],
        [7, 9],
        [6, 9],
    ]
    graph = [[] for i in range(11)]
    pgraph = [[] for i in range(11)]

    in_degree = [0 for i in range(11)]
    for e in edge:
        in_degree[e[1]] += 1
        graph[e[0]].append(e[1])
        pgraph[e[1]].append(e[0])

    entry_node = []

    for i in range(10):
        if in_degree[i] == 0:
            entry_node.append(i)
    if entry_node == []:
        return None
    virtual_entry = entry_node[0]
    if len(entry_node) > 1:
        for node in entry_node:
            in_degree[node] += 1
        virtual_entry = len(graph)
        graph.append([])
        graph[virtual_entry].extend(entry_node)

    queue = deque()
    current_group = []
    group_stack = [[]]
    sorted_groups = []

    def dfs(graph, node, visited):
        if node in visited:
            return
        visited.add(node)
        nonlocal current_group
        nonlocal group_stack
        nonlocal sorted_groups
        nonlocal queue
        current_group.append(node)

        if len(graph[node]) > 1 and current_group:
            group_stack[-1] = current_group
            current_group = []
            if not queue:
                sorted_groups.append(group_stack)
                group_stack = [[]]
            else:
                group_stack.append([])

        if queue:
            par_node = queue.popleft()
            dfs(graph, par_node, visited)

        for n in graph[node]:
            in_degree[n] -= 1
            if in_degree[n] == 0:

                if len(pgraph[n]) > 1 and group_stack:
                    group_stack[-1] = current_group
                    current_group = []
                    sorted_groups.append(group_stack)
                    group_stack = [[]]
                if len(graph[n]) > 1:
                    queue.append(n)
                else:
                    dfs(graph, n, visited)
            if current_group:
                group_stack[-1] = current_group
                group_stack.append([])
                current_group = []
        if queue:
            node = queue.popleft()
            dfs(graph, node, visited)

    visited = set()
    dfs(graph, virtual_entry, visited)
    if len(group_stack) > 1:
        if not group_stack[-1]:
            group_stack.pop()
        sorted_groups.append(group_stack)
        group_stack = [[]]
    for g in sorted_groups:
        print(g)
    return sorted_groups


if __name__ == "__main__":

    model_path = "../temp_host/lordtt13_emo_mobilebert.onnx"
    #model_path = "../temp_host/xlm-roberta-base.onnx"
    #model_path = "distilbert-base-uncased-finetuned-sst-2-english.onnx"
    testO_i = CaptureOnnxSubGraph()
    testO_i.load(model_path)
    testO_i.run()
