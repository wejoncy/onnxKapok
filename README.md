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

### ENv
Test by Python 3.7

1) install all python dependencies

    `pip install -r requirements.txt`

2) A model you want to compile in AOT, and run the main.py


## Capabilities or RoadMap

- [x] float model, usually bert-like models
- [x] element-wise ops
- [x] Reduction ops when reduction axes is [-1]
- [x] X86-64/arm64 with CPP module
- [ ] Triton backend
- [ ] int8/uint8 model



## Benchmarks

| model      | ort | or+aot |
| ----------- | ----------- |----|
| distilbert-base-cased-distilled-squad| 10.2ms| 9.2ms |
| lordtt13/emo-mobilebert|  15.2ms |   13.6ms |




