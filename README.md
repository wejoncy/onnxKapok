<link rel="stylesheet" type="text/css" href="docs/text/css/rdm.css" />

# onnx_igniter

## Why this repo?
There are too many rules about how to fuse nodes and a related Op has to be defined in ORT.

This repo is to extend fusion rule boundary in ORT and make it more flexible.
Basically, fusion nodes into a subgraph in a general way, then the subgraph can be replaced with a AOTOp node and compiled into a dynamic library.

A compiling process is in a AOT way. So you have to know where do you want to run the model.

Available backends are **x86-64/arm** with CPP backend and **Triton** backend.


## The basic idea
We will do it in two steps.

sub-graph-capture ---> sub-graph-compile

- graph-capture: To capture a subgraph from a model

    To fuse multiple nodes includes element-wise ,reduction, data-moving Ops into a subgraph.
We will make subgraph has more nodes as possible.
    In the process of graph-capture, the tools can detect cycle connections and always generated a valid subgraph.

- graph-compile: To compile a subgraph into a dynamic library
    A subgraph will be passed to codegen module according to different backends. CPP code will be generated if x86-64/arm64 backend is selected. Triton backend will generate a python code which typical follows triton code rules.


## how to use it?


### ENV
 Tested on
 -  Python 3.7+ 
 -  Ubuntu 20.04
 -  CMake 3.20 (used to build MIPP)
 -  GCC 11.1.0
 -  NDK 21.3.6528147(used for arm backend)


### Get start
1.  CLone this repo
    -  `git clone git@github.com:wejoncy/onnx_igniter.git`

    then
    -  `git submodule update --init --recursive`

    or
    -  `git clone --recursive git@github.com:wejoncy/onnx_igniter.git`

2.  install all python dependencies

    `pip install -r requirements.txt`

3.  start from run_benchmark

    `python benchmark/run_benchmark.py --model_name=distilbert-base-cased-distilled-squad --backend=x86_64`

    you will see logs like 
    ```
    benchmark model: >>> google/mobilebert-uncased >>> : time-cost changes from 12.4166ms to 9.56533ms, speedup: 22.96% 
    benchmark model: >>> csarron/mobilebert-uncased-squad-v2 >>> : time-cost changes from 8.37221ms to 5.95806ms, speedup: 28.84% 
    benchmark model: >>> lordtt13/emo-mobilebert >>> : time-cost changes from 21.4322ms to 14.0741ms, speedup: 34.33% 
    benchmark model: >>> xlm-roberta-base >>> : time-cost changes from 14.033ms to 11.831ms, speedup: 15.69%
    benchmark model: >>> distilbert-base-uncased >>> : time-cost changes from 6.64002ms to 5.6072ms, speedup: 15.55%
    ```


## Capabilities or RoadMap

- [x] float model, usually bert-like models
- [x] element-wise ops
- [x] Reduction ops when reduction axes is [-1]
- [x] X86-64/arm64 with CPP module
- [ ] Triton backend
- [ ] int8/uint8 model
- [ ] fp16 support



## Benchmarks
Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz

| model      | ort | or+aot |
| ----------- | ----------- |----|
| google/mobilebert-uncased|  12.4166ms | 9.56533ms|
| csarron/mobilebert-uncased-squad-v2|  8.37221ms | 5.95806ms|
| lordtt13/emo-mobilebert|  21.4322ms | 14.0741ms|
| xlm-roberta-base|  14.033ms | 11.831ms|
| distilbert-base-cased| 6.64002ms | 5.6072ms |



