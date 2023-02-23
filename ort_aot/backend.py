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
import utils
import cpufeature
import multiprocessing
import shutil

class GetVecLine(object):
    def __init__(self, compiler_func:callable):
        super().__init__()
        self.compiler_func = compiler_func
        self.func_name = "get_vec_line"
        self.func_handle = None

    def code_gen(self):
        dtype = 'float'
        code = f"""
#include <mipp/mipp.h>
extern "C" {{
int {self.func_name}() {{
    return mipp::N<{dtype}>();
}}

}}
"""
        return code
    
    def get_vec_line(self):
        if self.func_handle:
            return self.func_handle()

        with tempfile.NamedTemporaryFile() as lib_path:
            self.compiler_func(self.code_gen(), lib_path.name)
            import ctypes
            so = ctypes.CDLL(str(lib_path.name))
            self.func_handle = getattr(so, self.func_name)
            self.func_handle.restype = ctypes.c_int
            return self.func_handle()


class MainFunctionForDebug(Igniter_IR.IRNode):
    def __init__(self, function:Igniter_IR.FunctionNode):
        self.body = None
        self.func_name = function.name
        self.in_arg_type_shape = function.input
        self.out_arg_type_shape = function.output

    def code_gen(self, visitor: common.NodeVisitor, var_context: common.CodeGenContext, indent: int = 0):
        input_dtypes = [i.dtype for i in self.in_arg_type_shape]
        input_ctype = [common.NP_TYPE_C_TYPE[i.type] for i in input_dtypes]
        input_shapes = [i.shape.copy() for i in self.in_arg_type_shape]
        output_shapes = [i.shape.copy() for i in self.out_arg_type_shape]
        
        in_dynamic_shape_axis = [
            [idx for idx, i in enumerate(in_shape) if not i.is_number]
            for in_shape in input_shapes
        ]
        out_dynamic_shape_axis = [
            [idx for idx, i in enumerate(out_shape) if not i.is_number]
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
        import numpy as np
        numel_inputs = [np.prod(i) for i in input_shapes]
        numel_outputs = [np.prod(i) for i in output_shapes]
        
        max_dim = max([len(iv) for iv in in_dynamic_shape_axis])
        max_elem = max([np.prod(iv) for iv in input_shapes])
        idx = [ind for ind, iax in enumerate(
            in_dynamic_shape_axis) if len(iax) == (max_dim) and np.prod(input_shapes[ind]) == max_elem][0]

        in_dy_axis = in_dynamic_shape_axis[idx]
        input_shape = input_shapes[idx]
        

        code = f"""
#include <cassert>
int main(int argc, const char* argv[]) {{
    int n =0;
"""
        buf_read = []
        r_vars = [f'input{i}' for i in range(len(input_shapes))]
        for i in range(len(input_shapes)):
            buf_read.append(
                f"""
    FILE *fp{i}=fopen("a{i}.bin","rb");
    auto* input{i} = new {input_ctype[i]}[{numel_inputs[i]}];
    n = fread(input{i}, sizeof({input_ctype[i]}), {numel_inputs[i]}, fp{i});
    assert(n=={numel_inputs[i]});
    fclose(fp{i});
                """
            )
        buf_write = []
        w_vars =[f'output{i}' for i in range(len(output_shapes))]
        for i in range(len(output_shapes)):
            buf_write.append(
                f"""
    auto* output{i} = new float[{numel_outputs[i]}];
                """
            )
        code += "\n".join(buf_read)
        code += "\n".join(buf_write)
        code += f"""    
    const void* input_ptr[] = {{{', '.join(r_vars)}}};
    void* output_ptr[] = {{{', '.join(w_vars)}}};
    const int64_t shape_ptr[] = {{{input_shape[in_dy_axis[0]]},{input_shape[in_dy_axis[1]]}}};
    {self.func_name}(input_ptr, 0, {input_shape[0]*input_shape[1]},  shape_ptr,output_ptr);
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
        self.context: common.HardwareContext = self.init_context()

    def init_context(self):
        get_vec_line = GetVecLine(self.compile_to_so).get_vec_line
        return  common.HardwareContext(
            self.target, get_vec_line() if self.target=="x86_64" else 4)


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

    def compile_to_so(self, code: str, overwrite_lib_path:str=None):
        lib_path = overwrite_lib_path or self.lib_path
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


        cmake_args = ['-DWITH_DEBUG_EXE='+ ("ON" if debug_mode else "OFF")]#, '-G', 'Ninja']
        cmake_replace_rules = {'{EXTRA_CXX_FLAGS}': '-march=native' if target == "x86_64" else '',
                               '{TPT_DIR}': str(Path(".").resolve(strict=True))}

        if target == "x86_64":  # -march=native
            cmake_args += ['-DCMAKE_BUILD_TYPE=Release']
        elif target in ["arm64-v8a", "armeabi-v7a"]:
            assert os.getenv('ANDROID_NDK'), "Please set ANDROID_NDK environment variable"
            cmake_args += ['-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake',
            f'-DANDROID_ABI={target}',
            '-DANDROID_NATIVE_API_LEVEL=android-26',
            '-DANDROID_ARM_NEON=TRUE',
            '-DCMAKE_BUILD_TYPE=Release']
        else:
            raise Exception("not support target archtecure yet")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_dir = Path(tmpdirname)
            cmake_file = tmp_dir/"CMakeLists.txt"
            code_file = tmp_dir/"code.cc"
            build_dir = tmp_dir / "build"
            build_dir.mkdir()
            with open(code_file, "w") as f:
                f.write(code)
            with open(cmake_file, "w") as f:
                with open(Path(__file__).parent.parent / "template/CMakeLists.txt", "r") as f2:
                    cmake_template = f2.read()
                    for k,v in cmake_replace_rules.items():
                        cmake_template = cmake_template.replace(k, v)
                    f.write(cmake_template)

            with utils.DirContext(build_dir):
                out_str = subprocess.check_output(f'cmake ..  {" ".join(cmake_args)} ',shell=True).decode("utf-8")
                logger.debug(msg=f"cmake output: {out_str}")
                out_str = subprocess.check_output(f"make", shell=True).decode("utf-8")
                logger.debug(msg=f"make output: {out_str}")
                shutil.copyfile(build_dir/"libcode.so", lib_path)
                if debug_mode:
                    shutil.copyfile(build_dir/"code_exe", lib_path.with_suffix('').with_suffix('.exe'))
            return

        #with tempfile.TemporaryDirectory() as tmpdirname:
        #    code_file = os.path.join(tmpdirname, "code.cpp")
        #    with open(code_file, "w") as f:
        #        f.write(code)
        #    o_file = os.path.join(tmpdirname, "code.o")
        #    cmd = f"g++ {cxx_flag} -c {code_file}  -o {o_file}"
        #    out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")
        #    cmd = f"g++ -shared  {cxx_flag}  {o_file} -o {lib_path}  "
        #    out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")
#
        #    if debug_mode:
        #        cmd = f"{CXX} -g {code_file} -I{INC_FLAG} -o {lib_path}.exe"
        #        out_str = subprocess.check_output(cmd, shell=True).decode("utf-8")
        #    assert lib_path.exists(), "compile failed"

    def compile(self, models_with_name: dict):
        logger.info("vec_line  = " + str(self.context.vec_lanes))
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