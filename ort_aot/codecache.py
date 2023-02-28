# from torch/_inductor/codecache.py
import base64
import dataclasses
import functools
import getpass
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import types
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import cdll
from threading import Thread
from time import sleep, time
from typing import Any, Callable, Dict, List

import torch

# from torch.hub import _Faketqdm, tqdm
from torch.utils import cpp_extension
# from . import config, cuda_properties, exc

LOCK_TIMEOUT = 600

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0
_t0 = None


def _compile_start():
    global _t0
    if _t0 is None:
        _t0 = time()


def _compile_end():
    global _cumulative_compile_time, _t0
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)


log = logging.getLogger(__name__)


@functools.lru_cache(None)
def cache_dir():
    return os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}",
    )


class DiskCache:
    @staticmethod
    @functools.lru_cache(None)
    def _subdir():
        subdir = os.path.join(cache_dir(), "cached_tunings")
        os.makedirs(subdir, exist_ok=True)
        return subdir

    @staticmethod
    @functools.lru_cache(4096)
    def _read_file(path):
        with open(path, "r") as fd:
            return json.loads(fd.read())

    def __init__(self, unique_name):
        super().__init__()
        self.unique_name = unique_name

    def lookup(self, key: Any, generate: Callable[[], Any]):
        """
        Check if we have already generated key, if not call generate()
        to populate the cache.
        """
        path = os.path.join(self._subdir(), code_hash(self.unique_name + repr(key)))
        if not os.path.exists(path):
            value = generate()
            write_atomic(path, json.dumps(value))
        return self._read_file(path)


def get_lock_dir():
    lock_dir = os.path.join(cache_dir(), "locks")
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def code_hash(code):
    return (
        "c"
        + base64.b32encode(hashlib.sha256(code.encode("utf-8")).digest())[:51]
        .decode("utf-8")
        .lower()
    )


def get_code_path(source_code, ext, extra):
    basename = code_hash(source_code + extra)
    subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{ext}")
    return basename, subdir, path


def write(source_code, ext, extra=""):
    basename, subdir, path = get_code_path(source_code, ext, extra)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    if not os.path.exists(path):
        write_atomic(path, source_code)
    return basename, path


def write_atomic(path: str, source_code: str):
    # use a temp file for thread safety
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        f.write(source_code)
    os.rename(tmp_path, path)


class PyCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        key, path = write(source_code, "py")
        if key not in cls.cache:
            with open(path) as f:
                code = compile(f.read(), path, "exec")
                mod = types.ModuleType(f"{__name__}.{key}")
                mod.__file__ = path
                mod.key = key
                exec(code, mod.__dict__, mod.__dict__)
                # another thread might set this first
                cls.cache.setdefault(key, mod)
        return cls.cache[key]


@functools.lru_cache(None)
def patch_triton_dir():
    os.environ["TRITON_CACHE_DIR"] = os.environ.get(
        "TRITON_CACHE_DIR", os.path.join(cache_dir(), "triton")
    )


class TritonCodeCache:
    @staticmethod
    def get_name(mod):
        (name,) = [n for n in dir(mod) if n.startswith("triton_")]
        return name

    @classmethod
    def load(cls, source_code):
        patch_triton_dir()
        mod = PyCodeCache.load(source_code)
        return getattr(mod, cls.get_name(mod))


def _worker_compile(source_code, cc, device):
    cuda_properties.set_compiler_worker_current_device(device)
    kernel = TritonCodeCache.load(source_code)
    kernel.precompile(warm_cache_only_with_cc=cc)


def _load_kernel(source_code):
    kernel = TritonCodeCache.load(source_code)
    kernel.precompile()
    return kernel


def _load_kernel_name(source_code):
    return TritonCodeCache.get_name(PyCodeCache.load(source_code))
