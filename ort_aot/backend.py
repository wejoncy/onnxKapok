import common
import ir as Igniter_IR
from logger import logger
import codegen_cpu
import codegen_triton
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
            self.compiler_func(self.code_gen(), lib_path.name, ovewrite_debug=False)
            import ctypes
            so = ctypes.CDLL(str(lib_path.name))
            self.func_handle = getattr(so, self.func_name)
            self.func_handle.restype = ctypes.c_int
            return self.func_handle()


class CppBackend(object):
    def __init__(self, lib_path: Path, target: str, debug_mode=False):
        self.lib_path = lib_path
        self.target = target
        self.debug_mode = debug_mode
        self.context: common.HardwareContext = self.init_context()

    def init_context(self):
        get_vec_line = GetVecLine(self.compile_to_so).get_vec_line
        vec_lanes = get_vec_line() if self.target=="x86_64" else 4
        vec_lanes = 1024 if self.target=="triton" else vec_lanes
        return  common.HardwareContext(self.target, vec_lanes)


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

    def compile_to_so(self, code: str, overwrite_lib_path:str=None, ovewrite_debug=None):
        lib_path = overwrite_lib_path or self.lib_path
        target = self.target
        debug_mode = self.debug_mode if ovewrite_debug is None else ovewrite_debug

        if target == "triton":
            py_lib_path = lib_path.with_suffix(".py")
            with open(py_lib_path,'w') as fp:
                fp.write(code)
            return

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
        graph_lower = lowering.GraphLowering(self.target)
        module.lower(graph_lower, self.context)
        codegen_mod = codegen_cpu if self.target == "x86_64" else codegen_triton
        if debug_mode:
            # build a test with main function
            module.body.append(
                codegen_mod.MainFunctionForDebug(module.body[-1])
            )
        source_file_name = Path("gencode.cc")
        if self.target == "x86_64":
            visitor = codegen_mod.CPUCodeGen()
        else:
            visitor = codegen_mod.GPUCodeGen()
            source_file_name=source_file_name.with_suffix('').with_suffix('.py')
        src_code =module.code_gen(visitor, {})

        with open(source_file_name, "w") as f:
            f.write(src_code)

        self.compile_to_so(src_code)

        # print(src_code)
        return self.lib_path