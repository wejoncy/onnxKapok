from .backend import Backend
from . import codecache

import onnx
from typing import List, Tuple, Dict, Set
import types
import tempfile
from pathlib import Path

def replace_graph_input(model:onnx.ModelProto, input_shapes:List[List[int]]):
    """
    Replace the input shapes of the graph with deterministic shape.
    """
    graph = model.graph
    assert len(graph.input) == len(input_shapes)
    new_inputs = []
    for i, g_input in enumerate(graph.input):
        dtype = g_input.type.tensor_type.elem_type
        shape= input_shapes[i]
        tensor_type = onnx.helper.make_tensor_type_proto(elem_type=dtype, shape=shape)
        input_value = onnx.helper.make_value_info(g_input.name, tensor_type)
        new_inputs.append(input_value)
    for i in range(len(graph.input)):
        graph.input.pop()
    graph.input.extend(new_inputs)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def compile_for_ort_training(func_name:str, model:onnx.ModelProto, deterministic_input_shapes: List[List[int]])->types.ModuleType:
    model_with_name = {func_name: model}
    return compile_for_ort_training(model_with_name, deterministic_input_shapes)
    
def compile_for_ort_training(model_with_name:dict, deterministic_input_shapes: List[List[int]])->types.ModuleType:
    assert len(model_with_name) == 1, "Only one subgraph is supported"
    func_name, model = list(model_with_name.items())[0]
    model_with_name[func_name] = replace_graph_input(model, deterministic_input_shapes)
    target = "triton_ort_training"

    with tempfile.TemporaryDirectory() as tmpdirname:
        lib_path = Path(tmpdirname) / "lib.py"
        compiler_backend = Backend(lib_path, target, debug_mode=True)
        compiler_backend.compile(model_with_name)
        mod = codecache.PyCodeCache().load(lib_path.read_text())
        return mod
    raise RuntimeError("Failed to compile the model")

