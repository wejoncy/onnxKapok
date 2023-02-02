import common
import lower
from typing import Union, List, Tuple, Dict
from collections import defaultdict, deque,OrderedDict
import copy
import node_sets
import tempfile, os
from pathlib import Path
import subprocess
from sympy_utils import *

'''
DataType = {
    UNDEFINED = 0;
    // Basic types.
    FLOAT = 1;   // float
    UINT8 = 2;   // uint8_t
    INT8 = 3;    // int8_t
    UINT16 = 4;  // uint16_t
    INT16 = 5;   // int16_t
    INT32 = 6;   // int32_t
    INT64 = 7;   // int64_t
    STRING = 8;  // string
    BOOL = 9;    // bool

    // Advanced types
    FLOAT16 = 10;
    DOUBLE = 11;
    UINT32 = 12;
    UINT64 = 13;
    COMPLEX64 = 14;     // complex with float32 real and imaginary components
    COMPLEX128 = 15;    // complex with float64 real and imaginary components
    // Future extensions go here.
  }
AttributeType {
    UNDEFINED = 0;
    FLOAT = 1;
    INT = 2;
    STRING = 3;
    TENSOR = 4;
    GRAPH = 5;
    SPARSE_TENSOR = 11;
    TYPE_PROTO = 13;

    FLOATS = 6;
    INTS = 7;
    STRINGS = 8;
    TENSORS = 9;
    GRAPHS = 10;
    SPARSE_TENSORS = 12;
    TYPE_PROTOS = 14;
  }
'''

class Schedule(object):
    def __init__(self):
        pass

    def can_fusion_op(self, loop1: lower.Loop, loop2: lower.Loop):
        if (
            loop1.var != loop2.var
            or loop1.start != loop2.start
            or loop1.end != loop2.end
            or loop1.step != loop2.step
        ):
            return False
        return True

    def do_fusion_recursive(self, loop1: lower.Loop, loop2: lower.Loop):
        if not self.can_fusion_op(loop1, loop2):
            raise Exception("can not fusion loop")

        inner_loop1, inner_loop2 = loop1.body, loop2.body

        if loop1.depth < 2:
            if isinstance(inner_loop1, list):
                inner_loop1.append(inner_loop2)
                loop1.body = inner_loop1
            else:
                loop1.body = [inner_loop1, inner_loop2]
            return loop1

        loop1.body = self.do_fusion_recursive(inner_loop1, inner_loop2)
        return loop1

    def update_IO_after_fusion_op(
        self, bb1: lower.ExecutionBlock, bb2: lower.ExecutionBlock
    ):
        bb1.forward_var_set.update(bb2.forward_var_set)
        bb2.forward_var_set.clear()
        bb1.var_map.update(bb2.var_map)
        bb2.var_map.clear()
        bb1.intermediate_var.update(bb2.intermediate_var)
        bb2.intermediate_var.clear()
        bb1.load.update(bb2.load)
        bb2.load.clear()

        tmp_var_but_across_loop = OrderedDict()
        new_input = set(bb1.input)
        new_output = set(bb2.output + bb1.output)

        for var in bb2.input:
            if var in new_output:
                tmp_var_but_across_loop[var]=0
                new_output.remove(var)
            else:
                new_input.add(var)
                print(var.name)
        for i in tmp_var_but_across_loop.keys():
            if i.name not in bb1.forward_var_set:    
                bb1.forward_var_set[i.name]=i
        bb1.output = list(new_output)
        bb1.input = list(new_input)

    def fusion_op(self, blocks: List[lower.ExecutionBlock]):
        if len(blocks) < 2:
            return blocks
        de_blocks = deque(blocks)
        new_blocks = []
        while len(de_blocks) > 1:
            bb1: lower.ExecutionBlock = de_blocks.popleft()
            bb2: lower.ExecutionBlock = de_blocks.popleft()

            if not isinstance(bb1.body, lower.Loop):
                new_blocks.append(bb1)
                de_blocks.appendleft(bb2)
                continue
            if not isinstance(bb2.body, lower.Loop):
                new_blocks.append(bb1)
                new_blocks.append(bb2)
                continue
            if self.can_fusion_op(bb1.body, bb2.body):
                bb1.body = self.do_fusion_recursive(bb1.body, bb2.body)
                self.update_IO_after_fusion_op(bb1, bb2)

                de_blocks.appendleft(bb1)
            else:
                new_blocks.append(bb1)
                de_blocks.appendleft(bb2)
        new_blocks.extend(de_blocks)
        return new_blocks

    def tile_inner_loop(
        self, blocks: List[lower.ExecutionBlock], tile_size: int = 16
    ) -> List[lower.ExecutionBlock]:
        assert (
            len(blocks) == 1
        ), " only support one block now, but got {} blocks".format(len(blocks))
        bb1: lower.ExecutionBlock = blocks[0]
        if not isinstance(bb1.body, lower.Loop):
            return blocks
        sympy_factor = sympy.Integer(tile_size)

        def do_tile_loop(loop: lower.Loop):
            if loop.depth > 1:
                if isinstance(loop.body, list):
                    loop.body = [do_tile_loop(i) for i in loop.body]
                loop.body = do_tile_loop(loop.body)
                return loop

            if isinstance(loop.body, list):
                mutate_body = []
                for sub_body in loop.body:
                    if not isinstance(sub_body, lower.Loop):
                        mutate_body.append(sub_body)
                        continue
                    assert sub_body.depth == 0, "only support tile loop with depth 0"

                    main_loop: lower.Loop = sub_body

                    main_loop_range = IndexingDiv(
                        main_loop.end - main_loop.start, sympy_factor
                    )
                    offset = main_loop_range * sympy_factor

                    tail_loop = lower.Loop()
                    tail_loop.var = main_loop.var
                    tail_loop.start = offset
                    tail_loop.end = main_loop.end
                    tail_loop.body = copy.copy(main_loop.body)
                    tail_loop.attributes = main_loop.attributes

                    main_loop.end = main_loop.start + offset
                    if not (sympy.simplify(main_loop.end - main_loop.start) == 0):
                        mutate_body.append(main_loop)
                    if not (sympy.simplify(tail_loop.end - tail_loop.start) == 0):
                        mutate_body.append(tail_loop)
                loop.body = mutate_body
            return loop

        blocks[0].body = do_tile_loop(bb1.body)
        return blocks

    def parallelize_outer_loop(
        self, blocks: List[lower.ExecutionBlock], parallel_depth: int = 2
    ) -> List[lower.ExecutionBlock]:
        assert parallel_depth in [
            1,
            2,
        ], "only support parallelize outer loop with depth 1 or 2"
        assert (
            len(blocks) == 1
        ), " only support one block now, but got {} blocks".format(len(blocks))
        bb = blocks[0]
        assert isinstance(bb.body, lower.Loop), "only support parallelize outer loop"
        assert (
            bb.body.depth + 1 > parallel_depth
        ), "parallel depth should be smaller than loop depth"
        assert (
            bb.body.start == 0 and bb.body.step == 1
        ), "only support parallelize natural nest loop"

        bb.body.parallel = True

        if parallel_depth == 2:
            assert isinstance(
                bb.body.body, lower.Loop
            ), "only support parallelize outer loop"
            loop_depth_out_1 = bb.body
            loop_depth_out_2 = loop_depth_out_1.body
            assert (
                loop_depth_out_2.start == 0 and loop_depth_out_2.step == 1
            ), "only support parallelize natural nest loop"
            loop_depth_out_1.body = loop_depth_out_2.body
            loop_depth_out_1.parallel_nest_loop = loop_depth_out_2
        else:
            raise NotImplementedError(
                "only support parallelize outer loop with depth 1"
            )

        return blocks


class MainFunctionForDebug(lower.IRNode):
    def __init__(self, func_name: str, in_type_shape: list, out_type_shape: list):
        self.body = None
        self.func_name = func_name
        self.in_arg_type_shape = in_type_shape
        self.out_arg_type_shape = out_type_shape

    def code_spice(self, var_map: dict, indent: int = 0):
        in_shapes = [i[1] for i in self.in_arg_type_shape]
        out_shapes = [i[1] for i in self.out_arg_type_shape]
        in_dynamic_shape_axis = [
            [idx for idx, i in enumerate(in_shape) if isinstance(i, str)]
            for in_shape in in_shapes
        ]
        out_dynamic_shape_axis = [
            [idx for idx, i in enumerate(out_shape) if isinstance(i, str)]
            for out_shape in out_shapes
        ]
        assert (
            in_dynamic_shape_axis[0] == out_dynamic_shape_axis[0]
        ), "input and output dynamic shape axis should be same"
        assert len(in_dynamic_shape_axis[0]) in [
            1,
            2,3,
        ], "only support two dynamic shape axis"

        i_all_elem_s = []
        o_all_elem_s = []
        import numpy as np

        for input_shape, in_dy_axis in zip(in_shapes, in_dynamic_shape_axis):
            for dy in in_dy_axis:
                input_shape[dy] = 24
            if 0 in in_dy_axis:
                input_shape[0] = 1
            i_all_elem_s.append(np.prod(input_shape))

        for output_shape, out_dy_axis in zip(out_shapes, out_dynamic_shape_axis):
            for dy in out_dy_axis[1:]:
                output_shape[dy] = 24
            if 0 in out_dy_axis:
                output_shape[0] = 1 
            o_all_elem_s.append(np.prod(output_shape))

        code = f"""
#include <cassert>
int main(int argc, const char* argv[]) {{
    const char* input_file1 = "a0.bin";

    float* input1=0, *input2=0, *input3=0;
    input1 = new float[{i_all_elem_s[0]}];
    FILE *fp1=fopen("a0.bin","rb");
    int n =fread(input1, sizeof(float), {i_all_elem_s[0]}, fp1);
    assert(n=={i_all_elem_s[0]});
    fclose(fp1);
"""
        if len(i_all_elem_s) > 1:
            code += f"""
    FILE *fp2=fopen("a1.bin","rb");
    input2 = new float[{i_all_elem_s[1]}];
    n = fread(input2, sizeof(float), {i_all_elem_s[1]}, fp2);
    assert(n=={i_all_elem_s[1]});
    fclose(fp2);
"""
            if len(i_all_elem_s) > 2:
                code += f"""
    FILE *fp3=fopen("a2.bin","rb");
    input3 = new float[{i_all_elem_s[2]}];
    n = fread(input3, sizeof(float), {i_all_elem_s[2]}, fp3);
    assert(n=={i_all_elem_s[2]});
    fclose(fp3);
"""
            elif len(i_all_elem_s) > 3:
                raise Exception("not support more than 3 inputs yet")
        code += f"""
    float* output1=0, *output2=0;
    output1 = new float[{o_all_elem_s[0]}];
"""
        if len(o_all_elem_s) > 1:
            code += f"""
    output2 = new float[{o_all_elem_s[1]}];
"""
            if len(o_all_elem_s) > 2:
                raise Exception("not support more than 2 outputs yet")

        code += f"""    
    const float* input_ptr[] = {{input1,input2,input3}};
    float* output_ptr[] = {{output1,output2}};
    
    {self.func_name}(input_ptr, {len(i_all_elem_s)},0, {input_shape[0]*input_shape[1]},  {input_shape[in_dy_axis[0]]}, {input_shape[in_dy_axis[1]]},output_ptr);
    return 0;
}}  
        """
        return code


class CPPCodeGen(object):
    def __init__(self):
        pass
    
    def gen_cpp_code(
        self, module: lower.ModuleNode, global_buffer: common.GraphIOBuffer
    ):
        # generate function header

        code = ""
        if isinstance(module.body[-1], MainFunctionForDebug):
            code = "#include <cstdio>\n#include <cstdlib>\n"
        code_section = []
        
        code_section.append(module.code_spice({}, 0))
        

        code += "\n\n".join(code_section)

        return code


class CppBackend(object):
    def __init__(self):
        pass

    def lower(
        self,
        blocks: List[lower.ExecutionBlock],
        global_buffer: common.GraphIOBuffer,
        func_name: str,
    ) -> lower.FunctionNode:
        for block in blocks:
            block.lower()
        schedule = Schedule()
        blocks = schedule.fusion_op(blocks)
        blocks = schedule.tile_inner_loop(blocks)
        blocks = schedule.parallelize_outer_loop(blocks)
        func = lower.FunctionNode(
            global_buffer.var_buffer_in, global_buffer.var_buffer_out
        )
        func.body = blocks
        func.const_var = global_buffer.const_buffer
        func.name = func_name
        func.lower()
        return func

    def compile_to_so(self, code: str, lib_path: Path, target:str, Debug=False):
        if target == "x86_64":
            CXX = "g++"
        elif target == "aarch64":
            CXX = "/home/stcadmin/work/andriod/Android/Sdk/ndk/22.0.7026061/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android29 --gcc-toolchain=/home/stcadmin/work/andriod/Android/Sdk/ndk/22.0.7026061/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/stcadmin/work/andriod/Android/Sdk/ndk/22.0.7026061/toolchains/llvm/prebuilt/linux-x86_64/sysroot"
        else:
            raise Exception("not support target archtecure yet")
        with tempfile.TemporaryDirectory() as tmpdirname:
            code_file = os.path.join(tmpdirname, "code.cpp")
            with open(code_file, "w") as f:
                f.write(code)
            o_file = os.path.join(tmpdirname, "code.o")
            cmd = f"{CXX} -fPIC -O3 -c {code_file} -o {o_file}"
            out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")
            cmd = f"{CXX} -shared  -fPIC -O3 -o {lib_path} {o_file} "
            out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")

            if Debug:
                cmd = f"{CXX} -g {code_file} -o {lib_path}.exe"
                out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")
            assert lib_path.exists(), "compile failed"

    def compile(self, models_with_name: dict, lib_path:Path, target:str = "x86_64"):
        function_recipes = []
        DEBUG = True

        debug_idx = -1
        for func_name, model in models_with_name.items():
            debug_idx += 1
            # if debug_idx != 1:continue
            plan = ExecutionPrepare(model)
            plan.prepare()
            node_group = plan.create_execution_plan()
            function_recipe = (node_group, plan.external_buffer, func_name)
            function_recipes.append(function_recipe)

        module = lower.ModuleNode()
        module.lower(function_recipes, self.lower)

        if DEBUG:
            # build auto test main function
            in_arg = [i.name for i in plan.external_buffer.var_buffer_in]
            out_arg = [i.name for i in plan.external_buffer.var_buffer_out]
            in_type_shape = [plan.edge_graph.type_and_shape[i] for i in in_arg]
            out_type_shape = [plan.edge_graph.type_and_shape[i] for i in out_arg]
            module.body.append(
                MainFunctionForDebug(func_name, in_type_shape, out_type_shape)
            )

        codegen = CPPCodeGen()
        src_code = codegen.gen_cpp_code(
            module=module, global_buffer=plan.external_buffer
        )

        with open("code.cc", "w") as f:
            f.write(src_code)

        self.compile_to_so(src_code, lib_path, target, DEBUG)

        # print(src_code)
        return lib_path


class ExecutionPrepare(object):
    def __init__(self, model):
        self.edge_graph: ConnectionGraph = lower.ConnectionGraph(model)
        self.external_buffer: common.GraphIOBuffer = common.GraphIOBuffer()
        self.graph_io_name = set()

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

        reduce_nodes = node_sets.ReduceNodeSet(self.edge_graph.egraph.produced_by)
        sorted_nodes = []

        def has_non_reduce_child(queue):
            for n in queue:
                if n not in reduce_nodes:
                    return True
            return False

        while queue:
            node: Node = queue.popleft()
            if node.current_node in reduce_nodes and has_non_reduce_child(queue):
                queue.append(node)
                continue
            # print(node.current_node.name)
            if node.current_node.name not in self.graph_io_name:
                sorted_nodes.append(node)
            for n in node.output_nodes:
                in_degree[n] -= 1
                if in_degree[n] == 0:
                    queue.append(n)
        return sorted_nodes

    def analyze_io_buffer(self, groups):
        for i in self.edge_graph.model.graph.input:
            self.external_buffer.var_buffer_in.append(i)
        for i in self.edge_graph.model.graph.output:
            self.external_buffer.var_buffer_out.append(i)
        for i in self.edge_graph.model.graph.initializer:
            self.external_buffer.const_buffer.append(i)
        for i in self.edge_graph.constant_nodes.values():
            self.external_buffer.const_buffer.append(i)

        cached_buffer = OrderedDict()
        for g in groups:
            g.analyze_io(self.external_buffer, self.edge_graph, cached_buffer)

    def create_execution_plan(self):
        for i in self.edge_graph.model.graph.input:
            self.graph_io_name.add(i.name)
        for i in self.edge_graph.model.graph.output:
            self.graph_io_name.add(i.name)

        sorted_nodes = self.topological_with_reduce_last()

        intergroup_st = lower.InterGroupStrategy()
        node_group = intergroup_st.do_fusion(nodes=sorted_nodes)

        self.analyze_io_buffer(node_group)

        return node_group
