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
import copy

import sys

sys.path.append(str(Path(__file__).parent.resolve()))
import common
from backend import CppBackend
import node_sets
import graph_capture
from logger import logger
import utils


# target = "x86_64"
# target = "arm64-v8a"
def compile_model(
    model_path: Path, output_path: Path, lib_path: Path, target: str = "x86_64", ort_optimize_first: bool = False
):
    output_path.unlink(missing_ok=True)
    lib_path.unlink(missing_ok=True)
    capturer = graph_capture.CaptureOnnxSubGraph(ort_optimize_first)
    graph_lib_path= lib_path
    if target != "x86_64":graph_lib_path = Path(lib_path.name)
    model_with_name = capturer.run(model_path, graph_lib_path)
    cpp_backend = CppBackend(lib_path, target)
    cpp_backend.compile(model_with_name)

    # onnx.checker.check_model(self.model_proto)
    opset = capturer.model_proto.opset_import.add()
    opset.version = 1
    opset.domain = "com.microsoft"
    model_simp = capturer.model_proto
    # model_simp, check = simplify(self.model_proto)
    # assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    logger.info(f"successful compiled to onnx model:{output_path} lib_path:{lib_path}")
    return capturer.fused_node_nums


def debug_model(
    model_path: Path, output_path: Path, lib_path: Path, target: str = "x86_64", ort_optimize_first: bool = False
):
    capturer = graph_capture.CaptureOnnxSubGraph(ort_optimize_first)
    model_with_name = capturer.run(model_path, lib_path)
    cpp_backend = CppBackend(lib_path, target, debug_mode=True)
    model_with_name_copy = copy.deepcopy(model_with_name)
    cpp_backend.compile(model_with_name)

    test_lib(model_with_name_copy, lib_path)

    #for v in model_with_name.values():
    #    capturer.model_proto.graph.output.extend(v.graph.output)
    capturer.model_proto.graph.output.extend(
        [
            onnx.ValueInfoProto(
                name="/transformer/h.0/attn/Cast_1_output_0"
            )
        ]
    )

    opset = capturer.model_proto.opset_import.add()
    opset.version = 1
    opset.domain = "com.microsoft"
    onnx.save(capturer.model_proto, output_path)
    return capturer.fused_node_nums

class CostTime(object):
    def __init__(self, tcs: list, repeat: int = 1):
        self.st = 0
        self.tcs = tcs
        self.repeat = repeat

    def __enter__(self):
        self.st = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_time = time.perf_counter() - self.st
        self.tcs.append(total_time / self.repeat * 1000)


def test_lib(onnx_model_map: dict, lib_path: Path):
    if len(onnx_model_map) > 1:
        logger.info(" multi subgraphs detected, will test the last one")

    func_name, onnx_model = list(onnx_model_map.items())[-1]
    input_dtypes = [utils.get_elem_type_from_type_proto(inp.type) for inp in onnx_model.graph.input]
    output_dtypes = [utils.get_elem_type_from_type_proto(
        out.type) for out in onnx_model.graph.output]

    input_shapes = [utils.get_shape_from_value_info(inp) for inp in onnx_model.graph.input]
    output_shapes = [utils.get_shape_from_value_info(out) for out in onnx_model.graph.output]

    output_shapes[0][0] = 'batch'
    input_shapes[1][0] = 'batch'

    in_dynamic_shape_axis = [
        [idx for idx, i in enumerate(in_shape) if isinstance(i, str)]
        for in_shape in input_shapes
    ]
    out_dynamic_shape_axis = [
        [idx for idx, i in enumerate(out_shape) if isinstance(i, str)]
        for out_shape in output_shapes
    ]
    for input_shape, in_dy_axis in zip(input_shapes, in_dynamic_shape_axis):
        if input_shape == []:
            input_shape.append(1)
            continue
        for dy in in_dy_axis:
            input_shape[dy] = 24
        if 0 in in_dy_axis:
            input_shape[0] = 1

    for output_shape, out_dy_axis in zip(output_shapes, out_dynamic_shape_axis):
        for dy in out_dy_axis:
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
    for idx, input_shape in enumerate( input_shapes):
        if input_dtypes[idx] == 9:
            a_in.append(np.random.rand(*input_shape).astype(np.float32)>0.5)
        else:
            a_in.append(np.random.rand(*input_shape).astype(common.TENSOR_TYPE_TO_NP_TYPE[input_dtypes[idx]]))

    c_out = []
    for output_shape in output_shapes:
        c_out.append(np.zeros(shape=output_shape).astype(np.float32))

    input_count = len(input_shapes)
    output_count = len(output_shapes)
    in_arg_type = ctypes.c_void_p * input_count
    out_arg_type = ctypes.c_void_p * output_count
    shape_arg_type = ctypes.c_int

    if input_count == 2:
        in_arg = in_arg_type(a_in[0].ctypes.data, a_in[1].ctypes.data)
    elif input_count == 3:
        in_arg = in_arg_type(
            a_in[0].ctypes.data, a_in[1].ctypes.data, a_in[2].ctypes.data
        )
    elif input_count == 4:
        in_arg = in_arg_type(
            a_in[0].ctypes.data, a_in[1].ctypes.data, a_in[2].ctypes.data, a_in[3].ctypes.data
        ) 
    else:
        in_arg = in_arg_type(a_in[0].ctypes.data)

    if output_count == 2:
        out_arg = out_arg_type(c_out[0].ctypes.data, c_out[1].ctypes.data)
    else:
        out_arg = out_arg_type(c_out[0].ctypes.data)
    max_dim = max([len(iv) for iv in in_dynamic_shape_axis])
    max_elem = max([np.prod(iv) for iv in input_shapes])
    idx = [ind for ind, iax in enumerate(
        in_dynamic_shape_axis) if len(iax) == (max_dim) and np.prod(input_shapes[ind]) == max_elem][0]
    
    in_dy_axis = in_dynamic_shape_axis[idx]
    input_shape = input_shapes[idx]
    shape_arg_np = np.array([input_shape[in_dy_axis[0]], input_shape[in_dy_axis[1]]])
    shape_arg = shape_arg_np.ctypes.data

    c_tc = [0]
    with CostTime(c_tc) as tc:
        for i in range(10000):
            ret = func_handle(
                in_arg,
                0,
                input_shape[0] * input_shape[1],
                shape_arg,
                out_arg,
            )

    for idx, a_x in enumerate(a_in):
        a_x.tofile(f"a{idx}.bin")

    session = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_feed = {i.name: a_in[idx] for idx, i in enumerate(session.get_inputs())}

    py_tc = [0]
    with CostTime(py_tc) as tc:
        for i in range(10000):
            out = session.run([i.name for i in session.get_outputs()], input_feed)

    all_passed = all(
        [np.allclose(out[i], c_out[i], rtol=1e-03, atol=1e-05) for i in range(len(c_out))]
    )
    if all_passed:
        logger.info(f"Results are matched, time_cost: py_tc:{py_tc}, c_tc:{c_tc}")
    else:
        logger.warning("Results are not matched")


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
    model_path = Path("../temp_host/lordtt13_emo_mobilebert.onnx")
    model_path = Path("data/roberta_execution.onnx")
    # model_path = "../temp_host/xlm-roberta-base.onnx"
    # model_path = "distilbert-base-uncased-finetuned-sst-2-english.onnx"
    output_path = model_path.with_suffix('').with_suffix('._aot.onnx')
    lib_path = model_path.with_suffix('').with_suffix('.aot.so')
    debug_model(model_path, output_path, lib_path)


# TODO
'''
1. [x] better Symbolic Shape representation, especially for subgraph shape inference in ORT OP
2. [] re-compose ops like gelu
3. [] support data-movement ops, such as transpose, permute, slice gather in condition, and matmul
4. [] masked vectorization
'''