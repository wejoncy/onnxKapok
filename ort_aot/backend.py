import common
import ir as Igniter_IR
from logger import logger
import cpu
import triton
import lowering

from typing import Union, List, Tuple, Dict, Set
from collections import defaultdict, deque, OrderedDict
import copy
import tempfile, os
from pathlib import Path
import subprocess
from sympy_utils import *
import cpufeature
import multiprocessing


class MainFunctionForDebug(Igniter_IR.IRNode):
    def __init__(self, function:Igniter_IR.FunctionNode):
        self.body = None
        self.func_name = function.name
        self.in_arg_type_shape = function.input
        self.out_arg_type_shape = function.output

    def code_gen(self, visitor: common.NodeVisitor, var_context: common.CodeGenContext, indent: int = 0):
        return ''
        in_shapes = [i[1] for i in self.in_arg_type_shape]
        out_shapes = [i[1] for i in self.out_arg_type_shape]
        
        out_dynamic_shape_axis = [
            [idx for idx, i in enumerate(out_shape) if isinstance(i, str)]
            for out_shape in out_shapes
        ]
        out_dynamic_shape_symbol = [
            sp for out_shape in out_shapes for sp in out_shape if isinstance(sp, str)]
        
        in_dynamic_shape_axis = [
            [idx for idx, i in enumerate(in_shape) if isinstance(i, str)]
            for in_shape in in_shapes if in_shape
        ]
        
        assert (
            in_dynamic_shape_axis[0] == out_dynamic_shape_axis[0]
        ), "input and output dynamic shape axis should be same"
        assert len(in_dynamic_shape_axis[0]) in [
            1,
            2,
            3,
        ], "only support two dynamic shape axis"

        i_all_elem_s = []
        o_all_elem_s = []
        import numpy as np

        for input_shape, in_dy_axis in zip(in_shapes, in_dynamic_shape_axis):
            if input_shape == []:
                input_shape=[1]
                continue
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

        idx = [
            i
            for i in range(len(in_dynamic_shape_axis))
            if len(in_dynamic_shape_axis[i]) > 1
        ][0]
        in_dy_axis = in_dynamic_shape_axis[idx]
        input_shape = in_shapes[idx]

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

    def gen_cpp_code(self, module: Igniter_IR.ModuleNode):
        # generate function header

        code = ""
        if isinstance(module.body[-1], MainFunctionForDebug):
            code = "#include <cstdio>\n#include <cstdlib>\n"
        code_section = []
        visitor = cpu.CPUCodeGen()
        code_section.append(module.code_gen(visitor, {}))

        code += "\n\n".join(code_section)

        return code


class CppBackend(object):
    def __init__(self, lib_path: Path, target: str, debug_mode=False):
        self.lib_path = lib_path
        self.target = target
        self.debug_mode = debug_mode
        self.context: common.HardwareContext = common.HardwareContext(target, 16)

    def lower(
        self,
        blocks: List[Igniter_IR.ExecutionBlock],
        global_buffer: common.GraphIOBuffer,
        func_name: str,
        allow_vectorize: bool = True,
    ) -> Igniter_IR.FunctionNode:
        for block in blocks:
            block.lower()
        schedule = Schedule()
        blocks = schedule.fusion_op(blocks, set(i.name for i in global_buffer.var_buffer_out))
        blocks = schedule.tile_inner_loop(blocks, self.context.vec_lanes)
        if allow_vectorize:
            blocks = schedule.vectoring_inner_loop(blocks, self.context.vec_lanes)
        blocks = schedule.parallelize_outer_loop(blocks)
        func = Igniter_IR.FunctionNode(
            global_buffer.var_buffer_in, global_buffer.var_buffer_out
        )
        func.body = blocks
        func.const_var = global_buffer.const_buffer
        func.name = func_name
        func.hw_context = self.context
        func.lower()
        return func

    def get_simd_intrinsics(self):
        cpufeat = cpufeature.CPUFeature
        simd_flag = ""
        if cpufeat.get("AVX512f", False):
            simd_flag = "-mavx512f"
        elif cpufeat.get("AVX2", False):
            simd_flag = "-mavx2"

        if cpufeat.get("FMA3", False) or cpufeat.get("FMA4", False):
            simd_flag += " -mfma "
        return simd_flag

    def compile_to_so(self, code: str):
        lib_path = self.lib_path
        target = self.target
        debug_mode = self.debug_mode

        INC_FLAG = Path(".").resolve(strict=True) / "thirdparty/MIPP/install/include/"
        vec_flag = self.get_simd_intrinsics()
        # -fopt-info -fopenmp
        try_flag = (
            "-pipe -finline-functions -fomit-frame-pointer -fno-stack-protector "
            + " -fno-math-errno -fno-trapping-math -fno-common -fgraphite-identity "
            + " -floop-nest-optimize -ftree-loop-distribution "
            + " -fno-semantic-interposition -fipa-pta -fno-plt -ffast-math"
        )
        try_ld = "-Wl,--strip-all"
        cxx_flag = f"-std=c++17 -O3 {try_flag}  {try_ld} {vec_flag} -fPIC -I{INC_FLAG}"
        if target == "x86_64":
            CXX = "g++"
        elif target == "aarch64":
            CXX = (
                "/home/stcadmin/work/andriod/Android/Sdk/ndk/22.0.7026061/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ "
                + " --target=aarch64-none-linux-android29 "
                + " --gcc-toolchain=/home/stcadmin/work/andriod/Android/Sdk/ndk/22.0.7026061/toolchains/llvm/prebuilt/linux-x86_64 "
                + " --sysroot=/home/stcadmin/work/andriod/Android/Sdk/ndk/22.0.7026061/toolchains/llvm/prebuilt/linux-x86_64/sysroot"
            )
        else:
            raise Exception("not support target archtecure yet")
        with tempfile.TemporaryDirectory() as tmpdirname:
            code_file = os.path.join(tmpdirname, "code.cpp")
            with open(code_file, "w") as f:
                f.write(code)
            o_file = os.path.join(tmpdirname, "code.o")
            cmd = f"{CXX} {cxx_flag} -c {code_file}  -o {o_file}"
            out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")
            cmd = f"{CXX} -shared  {cxx_flag}  {o_file} -o {lib_path}  "
            out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")

            if debug_mode:
                cmd = f"{CXX} -g {code_file} -I{INC_FLAG} -o {lib_path}.exe"
                out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")
            assert lib_path.exists(), "compile failed"

    def compile(self, models_with_name: dict):
        debug_mode = self.debug_mode
        function_recipes = []

        module = Igniter_IR.ModuleNode(models_with_name)
        graph_lower = lowering.GraphLowering()
        module.lower(graph_lower, self.context)
        if debug_mode:
            # build a test with main function
            module.body.append(
                MainFunctionForDebug(module.body[-1])
            )

        codegen = CPPCodeGen()
        src_code = codegen.gen_cpp_code(module=module)

        with open("code.cc", "w") as f:
            f.write(src_code)

        self.compile_to_so(src_code)

        # print(src_code)
        return self.lib_path