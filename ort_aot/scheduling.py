from . import ir as Igniter_IR
from .logger import logger
from . import sympy_utils

import sympy
from typing import Union, List, Tuple, Dict, Set
from collections import defaultdict, deque, OrderedDict
import copy
from abc import ABCMeta, abstractmethod


class Schedule(object):
    def __init__(self):
        pass

    def can_fusion_op(self, loop1: Igniter_IR.Loop, loop2: Igniter_IR.Loop):
        if (
            loop1.var != loop2.var
            or loop1.start != loop2.start
            or loop1.end != loop2.end
            or loop1.step != loop2.step
        ):
            return False
        return True

    @abstractmethod
    def do_fusion_recursive(self, loop1: Igniter_IR.Loop, loop2: Igniter_IR.Loop):
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

    @abstractmethod
    def update_IO_after_fusion_op(
        self, bb1: Igniter_IR.ExecutionBlock, bb2: Igniter_IR.ExecutionBlock,
        global_input: Set[str], global_output: Set[str]
    ):
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
            if var in bb1.output and var.name not in global_output:
                tmp_var_but_across_loop[var.name] = var
                new_output.remove(var)
            elif var in bb1.input:
                pass
            elif var in global_input:
                # a var is not in bb1.input and not bb1.output
                new_input.add(var)
            else:
                bb2.forward_var_set[-1][var.name] = var

        for k, v in tmp_var_but_across_loop.items():
            if k not in bb1.forward_var_set[-1]:
                bb1.forward_var_set[-1][k] = v

        bb1.forward_var_set.extend(bb2.forward_var_set)
        bb1.output = list(new_output)
        bb1.input = list(new_input)

    @abstractmethod
    def fusion_op(self, blocks: List[Igniter_IR.ExecutionBlock], global_input: Set[str], global_output: Set[str]):
        if len(blocks) < 2:
            return blocks
        de_blocks = deque(blocks)
        new_blocks = []

        while len(de_blocks) > 1:
            bb1: Igniter_IR.ExecutionBlock = de_blocks.popleft()
            bb2: Igniter_IR.ExecutionBlock = de_blocks.popleft()

            if not isinstance(bb1.body, Igniter_IR.Loop):
                new_blocks.append(bb1)
                de_blocks.appendleft(bb2)
                continue
            if not isinstance(bb2.body, Igniter_IR.Loop):
                new_blocks.append(bb1)
                new_blocks.append(bb2)
                continue
            if self.can_fusion_op(bb1.body, bb2.body):
                bb1.body = self.do_fusion_recursive(bb1.body, bb2.body)
                bb1.fused_groups.append(bb2.group)
                self.update_IO_after_fusion_op(bb1, bb2, global_input, global_output)

                de_blocks.appendleft(bb1)
            else:
                new_blocks.append(bb1)
                de_blocks.appendleft(bb2)
        new_blocks.extend(de_blocks)
        return new_blocks

    @abstractmethod
    def tile_inner_loop(
        self, blocks: List[Igniter_IR.ExecutionBlock], tile_size: int = 16
    ) -> List[Igniter_IR.ExecutionBlock]:
        assert (
            len(blocks) == 1
        ), " only support one block now, but got {} blocks".format(len(blocks))
        bb1: Igniter_IR.ExecutionBlock = blocks[0]
        if not isinstance(bb1.body, Igniter_IR.Loop):
            return blocks
        sympy_factor = sympy_utils.sympy_symbol(tile_size)

        def do_tile_loop(loop: Igniter_IR.Loop):
            if loop.depth > 1:
                if isinstance(loop.body, list):
                    loop.body = [do_tile_loop(i) for i in loop.body]
                loop.body = do_tile_loop(loop.body)
                return loop

            list_loop = loop.body if isinstance(
                loop.body, list) else [loop.body]
            mutate_body = []
            for sub_body in list_loop:
                if not isinstance(sub_body, Igniter_IR.Loop):
                    mutate_body.append(sub_body)
                    continue
                assert sub_body.depth == 0, "only support tile loop with depth 0"

                main_loop: Igniter_IR.Loop = sub_body

                main_loop_range = sympy_utils.FloorDiv(
                    main_loop.end - main_loop.start, sympy_factor
                )
                offset = main_loop_range * sympy_factor

                tail_loop = Igniter_IR.Loop()
                tail_loop.var = main_loop.var
                tail_loop.start = offset
                tail_loop.end = main_loop.end
                tail_loop.body = copy.deepcopy(main_loop.body)
                tail_loop.attributes = main_loop.attributes

                main_loop.end = main_loop.start + offset
                pre_loop = None
                if not (sympy.simplify(main_loop.end - main_loop.start) == 0):
                    mutate_body.append(main_loop)
                    pre_loop = main_loop
                if not (sympy.simplify(tail_loop.end - tail_loop.start) == 0):
                    mutate_body.append(tail_loop)
                mutate_body.append(Igniter_IR.PostProcessBlock(pre_loop))
                mutate_body[-1].global_connections = bb1.connections
            loop.body = mutate_body
            return loop

        blocks[0].body = do_tile_loop(bb1.body)
        return blocks

    # only works for reduction && triton
    @abstractmethod
    def enable_recompute_for_reduction(self, blocks: List[Igniter_IR.ExecutionBlock]):
        return blocks

    @abstractmethod
    def vectoring_inner_loop(
        self, blocks: List[Igniter_IR.ExecutionBlock], lanes: int = 16
    ) -> List[Igniter_IR.ExecutionBlock]:
        assert (
            len(blocks) == 1
        ), " only support one block now, but got {} blocks".format(len(blocks))
        bb1: Igniter_IR.ExecutionBlock = blocks[0]
        if not isinstance(bb1.body, Igniter_IR.Loop):
            return blocks

        def do_vectoring_loop(loop: Igniter_IR.Loop):
            if loop.depth > 0:
                if isinstance(loop.body, list):
                    loop.body = [do_vectoring_loop(i) for i in loop.body]
                else:
                    loop.body = do_vectoring_loop(loop.body)
                return loop

            try_simplify = (loop.end - loop.start) % lanes
            if (
                loop.start != 0
                or not try_simplify.is_integer
                or (try_simplify.is_integer and try_simplify != 0)
                or loop.step != 1
                or loop.vectorization
                or loop.parallel
            ):
                loop.attributes = Igniter_IR.LoopAttr.ScalarLoop
                assert (
                    loop.body[0].vectorization == False
                ), "op in this loop should not be vectorized"
                return loop
            loop.vectorization = True
            loop.step = lanes
            for op in loop.body:
                assert isinstance(
                    op, Igniter_IR.IRNode
                ), f"expected op to be IRNode, but got {type(op)}"
                op.vectorization = True
            return loop

        blocks[0].body = do_vectoring_loop(bb1.body)
        return blocks

    @abstractmethod
    def parallelize_outer_loop(
        self, blocks: List[Igniter_IR.ExecutionBlock], parallel_depth: int = 2
    ) -> List[Igniter_IR.ExecutionBlock]:
        assert parallel_depth in [
            1,
            2,
        ], "only support parallelize outer loop with depth 1 or 2"
        assert (
            len(blocks) == 1
        ), " only support one block now, but got {} blocks".format(len(blocks))
        bb = blocks[0]
        assert isinstance(
            bb.body, Igniter_IR.Loop), "only support parallelize outer loop"

        # adjust parallel_depth if loop depth is smaller than parallel depth
        if bb.body.depth < parallel_depth and parallel_depth == 2:
            parallel_depth = 1
        if bb.body.depth < parallel_depth:
            return blocks
        assert (
            bb.body.depth + 1 > parallel_depth
        ), "parallel depth should be smaller than loop depth"
        assert (
            bb.body.start == 0 and bb.body.step == 1
        ), "only support parallelize natural nest loop"

        bb.body.parallel = True

        if parallel_depth == 2:
            assert isinstance(
                bb.body.body, Igniter_IR.Loop
            ), "only support parallelize outer loop"
            loop_depth_out_1 = bb.body
            loop_depth_out_2 = loop_depth_out_1.body
            assert (
                loop_depth_out_2.start == 0 and loop_depth_out_2.step == 1
            ), "only support parallelize natural nest loop"
            loop_depth_out_1.body = loop_depth_out_2.body
            loop_depth_out_1.parallel_nest_loop = loop_depth_out_2
        elif parallel_depth == 1:
            pass
        else:
            raise NotImplementedError(
                "only support parallelize outer loop with depth 1"
            )

        return blocks


class GPUSchedule(Schedule):
    def __init__(self):
        super().__init__()

    def tile_inner_loop(
        self, blocks: List[Igniter_IR.ExecutionBlock], tile_size: int = 16
    ) -> List[Igniter_IR.ExecutionBlock]:
        assert (
            len(blocks) == 1
        ), " only support one block now, but got {} blocks".format(len(blocks))
        bb1: Igniter_IR.ExecutionBlock = blocks[0]
        if not isinstance(bb1.body, Igniter_IR.Loop):
            return blocks
        sympy_factor = sympy_utils.sympy_symbol(tile_size)

        def do_tile_loop(loop: Igniter_IR.Loop):
            if loop.depth > 1:
                if isinstance(loop.body, list):
                    loop.body = [do_tile_loop(i) for i in loop.body]
                loop.body = do_tile_loop(loop.body)
                return loop

            list_loop = loop.body if isinstance(loop.body, list) else [loop.body]
            if len(list_loop) == 1:
                return loop
            mutate_body = []
            for sub_body in list_loop:
                if not isinstance(sub_body, Igniter_IR.Loop):
                    mutate_body.append(sub_body)
                    continue
                assert sub_body.depth == 0, "only support tile loop with depth 0"

                main_loop: Igniter_IR.Loop = sub_body
                mutate_body.append(main_loop)
                mutate_body.append(Igniter_IR.PostProcessBlock(main_loop))
                mutate_body[-1].global_connections = bb1.connections
            loop.body = mutate_body
            return loop

        blocks[0].body = do_tile_loop(bb1.body)
        return blocks

    @staticmethod
    def get_all_inputs_need_recompute(loop: Igniter_IR.Loop, graph_inputs: List[Igniter_IR.ComputeBuffer],
                                      graph_outputs: List[Igniter_IR.ComputeBuffer]):
        # order is not important
        inputs_set = set()
        outputs_set = set()
        for op in loop.body:
            for i in op.input:
                inputs_set.add(i)
            for o in op.output:
                outputs_set.add(o)
        extern_inputs = []
        # 1. input is not a scalar
        # 2. input is not in graph_inputs
        for inp in (inputs_set - outputs_set):
            if inp.predecessor is None or inp.shape[-1] == 1 or inp in graph_inputs or inp in graph_outputs:
                continue

            extern_inputs.append(inp)

        return extern_inputs

    @staticmethod
    def collect_predecessors_for_recompute(inputs: List[Igniter_IR.ComputeBuffer],
                                           graph_inputs: List[Igniter_IR.ComputeBuffer],
                                           graph_outputs: List[Igniter_IR.ComputeBuffer], load_map: Dict,
                                           forward_var_set: Dict[str, Igniter_IR.ComputeBuffer]):
        # order is not important
        inputs_set = set()
        for inp in inputs:
            if inp.predecessor is None or inp.shape[-1] == 1:
                forward_var_set[inp.name] = inp
                continue
            if inp in graph_inputs or inp in graph_outputs:
                reused_loadnode = load_map[inp.name]
                inputs_set.add(reused_loadnode)
                continue

            inputs_set.add(inp.predecessor)
            inputs_set.update(GPUSchedule.collect_predecessors_for_recompute(
                inp.predecessor.input, graph_inputs, graph_outputs, load_map, forward_var_set))

        return inputs_set

    # when reduction op has a lot processor ops, and the intermediate value has to be reused by many processor ops
    # If we have enough shared memory, we can easily cache it in shared memory.
    # However, Shared memory is always limited, so we need to recompute it when it is not cached in shared memory.
    # It's trading less kernel launch with more computation.
    # Could we use global memory as a cache? Not sure if triton supports it.
    def enable_recompute_for_reduction(self, blocks: List[Igniter_IR.ExecutionBlock]):
        assert (
            len(blocks) == 1
        ), " only support one block now, but got {} blocks".format(len(blocks))
        bb1: Igniter_IR.ExecutionBlock = blocks[0]
        if not isinstance(bb1.body, Igniter_IR.Loop):
            return blocks

        def do_recompute(top_loop: Igniter_IR.Loop):
            if top_loop.depth > 1:
                if isinstance(top_loop.body, list):
                    top_loop.body = [do_recompute(i) for i in top_loop.body]
                else:
                    top_loop.body = do_recompute(top_loop.body)
                return top_loop

            list_loop: List[Igniter_IR.Loop] = top_loop.body if isinstance(top_loop.body, list) else [top_loop.body]
            if len(list_loop) == 1:
                return top_loop

            load_map = {}
            map_op_index = {}
            idx = 0
            for loop in list_loop:
                for op in loop.body:
                    map_op_index[op] = idx
                    idx += 1

                    if isinstance(op, Igniter_IR.MaskLoadNode):
                        load_map[op.input[0].name] = op

            for lidx, cur_loop in enumerate(reversed(list_loop)):
                cur_loop.recompute = True
                bb1.recompute = True
                # remove unused store-node as we are going to recompute it
                body_len = len(cur_loop.body)
                for idx, op in enumerate(reversed(cur_loop.body.copy())):
                    if isinstance(op, Igniter_IR.MaskStoreNode):
                        if op.input[0] not in bb1.output:
                            cur_loop.body.pop(body_len-1-idx)
                    elif isinstance(op, Igniter_IR.MaskLoadNode):
                        if op.input[0] not in bb1.input and op.input[0] not in bb1.output:
                            cur_loop.body.pop(body_len-1-idx)

                input_need_recompute = self.get_all_inputs_need_recompute(cur_loop, bb1.input, bb1.output)
                if not input_need_recompute:
                    continue
                lidx = len(list_loop)-1-lidx-1
                prev_forward_map = list_loop[lidx].forward_var_set if lidx >= 0 else {}
                nodes_for_recompute = self.collect_predecessors_for_recompute(
                    input_need_recompute, bb1.input, bb1.output, load_map, prev_forward_map)
                nodes_for_recompute = list(nodes_for_recompute)
                sorted_nodes_for_recompute = sorted(nodes_for_recompute, key=lambda x: map_op_index[x])
                cur_loop.body = sorted_nodes_for_recompute + cur_loop.body


            return top_loop

        blocks[0].body = do_recompute(bb1.body)
        return blocks

    def vectoring_inner_loop(
        self, blocks: List[Igniter_IR.ExecutionBlock], lanes: int = 1024
    ) -> List[Igniter_IR.ExecutionBlock]:
        assert (len(blocks) == 1), " only support one block now, but got {} blocks".format(len(blocks))
        bb1: Igniter_IR.ExecutionBlock = blocks[0]
        if not isinstance(bb1.body, Igniter_IR.Loop):
            return blocks

        def do_vectoring_loop(loop: Igniter_IR.Loop):
            if loop.depth > 0:
                if isinstance(loop.body, list):
                    loop.body = [do_vectoring_loop(i) for i in loop.body]
                else:
                    loop.body = do_vectoring_loop(loop.body)
                return loop

            try_simplify = (loop.end - loop.start) > lanes
            if (
                loop.start != 0
                or loop.step != 1
                or loop.vectorization
                or loop.parallel
            ):
                loop.attributes = Igniter_IR.LoopAttr.ScalarLoop
                assert (loop.body[0].vectorization == False), "op in this loop should not be vectorized"
                return loop
            loop.vectorization = True
            if try_simplify.is_Boolean and try_simplify:
                loop.step = sympy_utils.sympy_symbol(name="RBLOCK")
            else:
                loop.step =  loop.end
            for op in loop.body:
                assert isinstance(op, Igniter_IR.IRNode), f"expected op to be IRNode, but got {type(op)}"
                op.vectorization = True
            return loop

        blocks[0].body = do_vectoring_loop(bb1.body)
        return blocks

    def parallelize_outer_loop(
        self, blocks: List[Igniter_IR.ExecutionBlock]
    ) -> List[Igniter_IR.ExecutionBlock]:
        assert (
            len(blocks) == 1
        ), " only support one block now, but got {} blocks".format(len(blocks))
        bb = blocks[0]
        assert isinstance(bb.body, Igniter_IR.Loop), "only support parallelize outer loop"
        parallel_depth = bb.body.depth
        if parallel_depth < 1:
            return blocks
        assert (bb.body.start == 0 and bb.body.step == 1), "only support parallelize natural nest loop"

        bb.body.parallel = True

        if parallel_depth >= 2:
            this_loop: Igniter_IR.Loop = bb.body
            nest_loops = []
            while this_loop.depth > 1:
                assert isinstance(this_loop.body, Igniter_IR.Loop), "only support parallelize outer loop"
                assert (this_loop.body.start == 0 and this_loop.body.step ==
                        1), "only support parallelize natural nest loop"
                nest_loops.append(this_loop.body)
                this_loop = this_loop.body

            loop_depth_out_1: Igniter_IR.Loop = bb.body
            loop_depth_out_1.body = nest_loops[-1].body
            loop_depth_out_1.parallel_nest_loop = nest_loops
        elif parallel_depth == 1:
            pass
        else:
            raise NotImplementedError("only support parallelize outer loop with depth 1")

        return blocks
