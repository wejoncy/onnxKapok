import os
import sys
from pathlib import Path
import argparse
from run import run
import cmake

def run_subprocess(
    args,
    cwd=None,
    capture_stdout=False,
    dll_path=None,
    shell=False,
    env={},
    python_path=None,
    cuda_home=None,
):
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            if "PATH" in my_env:
                my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
            else:
                my_env["PATH"] = dll_path
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path
    # Add nvcc's folder to PATH env so that our cmake file can find nvcc
    if cuda_home:
        my_env["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + my_env["PATH"]

    if python_path:
        if "PYTHONPATH" in my_env:
            my_env["PYTHONPATH"] += os.pathsep + python_path
        else:
            my_env["PYTHONPATH"] = python_path

    my_env.update(env)

    return run(*args, cwd=cwd, capture_stdout=capture_stdout, shell=shell, env=my_env)


def update_submodules(source_dir):
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=source_dir)
    run_subprocess(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir
    )

class DirContext(object):
    def __init__(self, build_dir: Path):
        self.cur_dir = None
        self.build_dir = build_dir
    def __enter__(self):
        self.cur_dir = Path('.').resolve()
        os.chdir(path=self.build_dir)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(path=self.cur_dir)
        
        
def build_mipp(android_abi=None, android_ndk_path=None):
    mipp_path = Path(__file__).parent.parent.parent.joinpath("thirdparty/MIPP").resolve(strict=True)

    mipp_install_path = mipp_path/"install"
    mipp_install_path.mkdir(parents=False, exist_ok=True)
    
    cmd_args = ["cmake", ".."]
    if android_abi:
        mipp_build_path = mipp_path/"android_build"
        mipp_build_path.mkdir(parents=False, exist_ok=True)
        cmd_args.append(f'-DANDROID_ABI="{android_abi}"')
        cmd_args.append(f'-DANDROID_NDK="{android_ndk_path}"')
        cmd_args.append(f'-DANDROID_PLATFORM="android-22"')
        cmd_args.append(f'-DCMAKE_TOOLCHAIN_FILE="{android_ndk_path}/build/cmake/android.toolchain.cmake"')
    else:
        mipp_build_path = mipp_path/"build"
        mipp_build_path.mkdir(parents=False, exist_ok=True)
        
    print(f"mipp_build_path: {mipp_build_path}")
    print(f"mipp_install_path: {mipp_install_path}")
    
    with DirContext(mipp_build_path):
        run_subprocess(cmd_args)
        run_subprocess(["cmake", "--install", ".", "--prefix", str(mipp_install_path)])


def main():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""
        build onnx_igniter
        """,
    )

    parser.add_argument(
        "-k",
        "--android_ndk_path",
        type=Path,
        required=False,
        help="NDK path.",
    )

    parser.add_argument(
        "-a",
        "--android_abi",
        type=str,
        required=False,
        choices=["armeabi-v7a", "arm64-v8a", "x86_64"],
        help="If build for android, default Linux x86_64.",
    )

    parser.add_argument(
        "-t",
        "--use_triton",
        action="store_true",
        help="default false,.",
    )

    parser.add_argument(
        "-c",
        "--use_cpp",
        action="store_false",
        help="default true.",
    )

    args = parser.parse_args()
    if not args.android_abi:
        print(" build for LInux x86_64")
    else:
        if not args.android_ndk_path:
            args.android_ndk_path = Path(
                os.getenv(key="ANDROID_NDK_HOME", default=None)
            )
        print(
            f"build for android {args.android_abi}, with NDK : {args.android_ndk_path}"
        )
    # test cmake
    assert cmake.CMAKE_BIN_DIR, "cmake not found"
    build_mipp(args.android_abi, args.android_ndk_path)


if __name__ == "__main__":
    main()
