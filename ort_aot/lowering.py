import common
import ir
import execution_planer
import node_sets
import scheduling
import sympy_utils
import utils

from typing import Union, List, Tuple, Dict, Set
from collections import defaultdict
from functools import wraps


def Singleton(cls):
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return getinstance


@Singleton
class UniqueNameGenerator(object):
    def __init__(self):
        self.count = 0

    def get_unique_var_name(self, prefix):
        self.count += 1
        return prefix + str(self.count)


def create_load_or_store(buf: ir.ComputeBuffer, is_load: bool, target: str):
    if target != "triton":
        if is_load:
            return [ir.LoadNode(buf)]
        else:
            return [ir.StoreNode(buf)]
    else:
        # range_id_name = UniqueNameGenerator().get_unique_var_name("triton_range_")
        # range_buf = ir.ComputeBuffer(range_id_name, shape=buf.shape[-1:])
        # range_ir = ir.RangeNode(0, buf.shape[-1], 1, range_buf)
        #
        # add_id_name = UniqueNameGenerator().get_unique_var_name("triton_add_")
        # add_out = ir.ComputeBuffer(add_id_name, shape=buf.shape[-1:])
        # addr_ir = ir.ComputeNode("Add", [buf, range_buf], [add_out], add_id_name+"_add")
        #
        mask_id_name = UniqueNameGenerator().get_unique_var_name("triton_mask_")
        mask_buf = ir.ComputeBuffer(mask_id_name, shape=buf.shape[-1:])
        # mask_ir = ir.MaskNode([range_buf, "vec_loop_step"], mask_buf)

        if is_load:
            return [ir.MaskLoadNode(buf, mask_buf)]
        else:
            return [ir.MaskStoreNode(buf, mask_buf)]


def insert_load_and_store(
    block: ir.ExecutionBlock, global_buffer: common.GraphIOBuffer,
    c_graph: execution_planer.ConnectionGraph, target: str = "x86_64"
):
    input_name_map = {inp.name: inp for inp in block.input}
    output_name_map = {inp.name: inp for inp in block.output}
    new_group = []
    # avoid duplicate load
    load_cache = set()
    for g in block.group:
        for inp in g.input:
            producer_op = (
                c_graph.egraph.produced_by[inp][0]
                if inp in c_graph.egraph.produced_by
                else None
            )

            if (
                inp in block.load or inp in input_name_map
            ) and producer_op not in node_sets.ReduceNodeSetInternal():
                load_buf = block.load[inp] if inp in block.load else input_name_map[inp]
                # we just skip unused load for constant scalar
                if (load_buf.data is not None and load_buf.data.size == 1) or load_buf.name in load_cache:
                    continue
                load_cache.add(load_buf.name)
                new_group.extend(create_load_or_store(load_buf, True, target))
        new_group.append(g)
        for out in g.output:
            if out in output_name_map and not isinstance(g, ir.ReduceNode):
                load_cache.add(out)
                new_group.extend(create_load_or_store(output_name_map[out], False, target))

    block.group = new_group


def analyze_io(
    block: ir.ExecutionBlock,
    global_buffer: common.GraphIOBuffer,
    c_graph: execution_planer.ConnectionGraph,
    cached_buffer: Dict[str, ir.ComputeBuffer],
    target: str = "x86_64"
):
    # should we keep buffer here?
    # self.cached_buffer = cached_buffer
    inputs = defaultdict(lambda: 0)
    outputs = defaultdict(lambda: 0)
    loads = set()
    const_buffer_set = set([i.name for i in global_buffer.const_buffer])
    external_buffer_out_set = set(
        [i.name for i in global_buffer.var_buffer_out])

    def is_const_input(inp):
        return inp in const_buffer_set or (
            inp in c_graph.egraph.produced_by
            and c_graph.egraph.produced_by[inp][0].op_type == "Constant"
        )

    for g in block.group:
        for inp_bf in g.input:
            assert isinstance(inp_bf, ir.ComputeBuffer)
            inp = inp_bf.name
            if not is_const_input(inp):
                inputs[inp] += 1
            else:
                loads.add(inp_bf)
        for out_b in g.output:
            assert isinstance(out_b, ir.ComputeBuffer)
            out = out_b.name
            if out not in const_buffer_set:
                outputs[out] = 1
            else:
                raise Exception("const buffer can not be output")
    # self.intermediate_var  = inputs.intersection(outputs)

    for out_name in list(outputs.keys()):
        if out_name in list(inputs.keys()) and out_name not in external_buffer_out_set:
            outputs[out_name] = len(
                c_graph.egraph.consumed_by[out_name]) - inputs[out_name]
            block.intermediate_var[out_name] = 0
            inputs.pop(out_name)
            assert outputs[out_name] >= 0, "output buffer can not be input"
            if outputs[out_name] == 0:
                outputs.pop(out_name)

    for v in outputs:
        assert v in cached_buffer, "found unhandled output buffer!!!"
        buffer = cached_buffer[v]
        if v not in c_graph.egraph.graph_output_names:
            buffer.attr_cross_loop = True
        block.output.append(buffer)

    for v in inputs:
        assert v in cached_buffer, "found unhandled output buffer!!!"
        buffer = cached_buffer[v]
        if v not in c_graph.egraph.graph_input_names and v not in c_graph.egraph.graph_output_names:
            buffer.attr_cross_loop = True
        block.input.append(buffer)

    for ov in loads:
        v = ov

        if v.name in c_graph.egraph.produced_by:
            pv = c_graph.egraph.produced_by[v][0].name
        else:
            pv = v.name
        tv = c_graph.constant_nodes[pv]
        data = common.parse_onnx_to_numpyarray(tv)
        assert (data == v.data).all(), "const buffer not matched"
        buffer = v
        cached_buffer[v] = buffer
        block.load[ov] = buffer

    insert_load_and_store(block, global_buffer, c_graph, target)
    pass


class GraphLowering(common.NodeVisitor):
    def __init__(self, target: str):
        super().__init__()
        self.target = target

    def visit(self, node: ir.IRNode, context: common.HardwareContext, indent: int = 0):
        fn = getattr(self, node.__class__.__name__)
        assert fn is not None, "unimplemented node: %s" % node.__class__.__name__
        return fn(node, context)

    def FunctionNode(self, node: ir.FunctionNode, context: common.HardwareContext):
        assert len(node.body) == 1, "multiple body not supported in function node"
        shape_var = [i for i in node.body[0].shape if i.is_symbol]
        node.shape_var = list(set(shape_var))
        node.shape_var.sort(key=shape_var.index)

        node.body[0].gen_var(node.const_var, self.target == "triton")
        node.body[0].analyze_io_connections()

    def ModuleNode(self, node: ir.ModuleNode, context: common.HardwareContext):
        allow_vectorize = True

        def lower_to_functionNode(blocks: List[ir.ExecutionBlock],
                                  global_buffer: common.GraphIOBuffer,
                                  func_name: str,
                                  allow_vectorize: bool):
            for block in blocks:
                block.lower(self, context)
            if self.target == 'triton':
                schedule = scheduling.GPUSchedule()
            else:
                schedule = scheduling.Schedule()
            blocks = schedule.fusion_op(blocks, set(i.name for i in global_buffer.var_buffer_in),
                                        set(i.name for i in global_buffer.var_buffer_out))
            blocks = schedule.tile_inner_loop(blocks, context.vec_lanes)
            if allow_vectorize or self.target == 'triton':
                blocks = schedule.vectoring_inner_loop(blocks, context.vec_lanes)
            blocks = schedule.parallelize_outer_loop(blocks)
            func = ir.FunctionNode(global_buffer.var_buffer_in, global_buffer.var_buffer_out)
            func.body = blocks
            func.const_var = global_buffer.const_buffer
            func.name = func_name
            func.hw_context = context
            func.lower(self, context)
            return func

        for idx, (func_name, model) in enumerate(node.modules.items()):
            plan = execution_planer.ExecutionPrepare(model, self.target)
            plan.prepare()
            node_group = plan.create_execution_plan(analyze_io)
            function: FunctionNode = lower_to_functionNode(
                node_group, plan.external_buffer, func_name, allow_vectorize)
            node.body.append(function)
        node.has_vectorization = allow_vectorize

    def ExecutionBlock(self, node: ir.ExecutionBlock, context: common.HardwareContext):
        # add Loop()
        node.body = node.build_loop()
