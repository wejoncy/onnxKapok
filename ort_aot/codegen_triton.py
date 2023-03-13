from .ir import *
from . import common


def _get_type(t):
    out_dtype = common.NP_TYPE_C_TYPE[t.type]
    return out_dtype


class MainFunctionForDebug(IRNode):
    def __init__(self, func: FunctionNode):
        super().__init__()
        # self.func = func
        self.dynamic_shape = func.shape_var
        self.func_name = func.name
        self.in_arg_type_shape = func.input
        self.out_arg_type_shape = func.output

    def create_wrapper(self):
        input_dtypes = [i.dtype for i in self.in_arg_type_shape]
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

        used_shape = [1, 128]
        for input_shape, in_dy_axis in zip(input_shapes, in_dynamic_shape_axis):
            if input_shape == []:
                input_shape.append(1)
                continue
            for dy in in_dy_axis:
                input_shape[dy] = used_shape[1]
            if 0 in in_dy_axis:
                input_shape[0] = 1

        for output_shape, out_dy_axis in zip(output_shapes, out_dynamic_shape_axis):
            for dy in out_dy_axis:
                output_shape[dy] = used_shape[1]
            if 0 in out_dy_axis:
                output_shape[0] = 1

        need_indent = " " * 4

        shape_define = [need_indent + f"{sp}= {used_shape[idx%2]}\n" for idx, sp in enumerate(self.dynamic_shape)]
        shape_define = ''.join(shape_define)
        dynamic_var = [str(vsp) for vsp in self.dynamic_shape]
        dynamic_var = ','.join(dynamic_var)
        if dynamic_var:
            dynamic_var += ","

        input_tensors_expr = [
            need_indent + f"input_{i} = torch.rand({in_shape}, device='cuda')\n" for i, in_shape in enumerate(input_shapes)]
        input_tensors_expr = ''.join(input_tensors_expr)
        input_tensors_var = [f"input_{i}" for i, t in enumerate(self.in_arg_type_shape)]
        input_tensors_var = ','.join(input_tensors_var)

        output_tensors_expr = [
            need_indent + f"output_{i} = torch.empty({out_shape}, device='cuda')\n" for i, out_shape in enumerate(output_shapes)]
        output_tensors_expr = ''.join(output_tensors_expr)
        output_tensors_var = [f"output_{i}" for i, t in enumerate(self.out_arg_type_shape)]
        output_tensors_var = ','.join(output_tensors_var)

        src = ""
        src += f"""

import torch
if __name__ == "__main__":
{shape_define}

{input_tensors_expr}
{output_tensors_expr}

    n_elements_in_last_dim = output_0.shape[-1]
    paralleled_blocks = output_0.numel()//n_elements_in_last_dim
    grid = lambda meta: (paralleled_blocks* triton.cdiv(n_elements_in_last_dim, meta['RBLOCK']),)
    {self.func_name}[grid]({input_tensors_var}, {output_tensors_var}, {dynamic_var} RBLOCK=triton.next_power_of_2(n_elements_in_last_dim))
    print(output_0[0,0,0,:10])
    """
        return src


class GPUCodeGen(common.NodeVisitor):
    def __init__(self):
        super().__init__()

    def visit(self, node: IRNode, context: common.CodeGenContext, indent: int):
        fn = getattr(self, node.__class__.__name__)
        assert fn is not None, "unimplemented node: %s" % node.__class__.__name__
        return fn(node, context, indent)

    def MainFunctionForDebug(self, node: MainFunctionForDebug, context: common.CodeGenContext, indent: int):
        return node.create_wrapper()

    def Loop(self, node: Loop, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        vec_var_set = var_context.vectorized_var_set
        need_indent = " " * indent
        dec_header = ""
        # forward declaration
        for fvar, buffer_l in node.forward_var_set.items():
            if node.step == node.end:
               break
            buffer = buffer_l[0] if isinstance(buffer_l, list) else buffer_l
            str_var = str(fvar)
            if buffer.shape is not None and buffer.shape[-1] == 1:
                if str_var not in node.reduction_var:
                    init_val = 0.0
                elif "sum" in node.reduction_var[str_var].lower():
                    init_val = 0.0
                elif "max" in node.reduction_var[str_var].lower():
                    init_val = "-3.40082e38"
                elif "min" in node.reduction_var[str_var].lower():
                    init_val = "3.40082e38"
                else:
                    assert False, ("unsupported reduction type: %s" % node.reduction_var[str_var])

                dec_header += need_indent + f"{var_map[str_var]} = {init_val}\n"
                if node.vectorization:
                    dtype = _get_type(buffer.dtype)
                    dec_header += need_indent + \
                        f"vec_{var_map[str_var]} = tl.zeros([RBLOCK], dtype=tl.{buffer.dtype.name})\n"
                    vec_var_set.add(var_map[str_var])
                    node.var_need_post_process[str_var] = f"vec_{var_map[str_var]}"
            else:
                assert False, "buffer should be defined in the ExecutionBlock"
                dec_header += (
                    need_indent
                    + f"float e_{var_map[str_var]}[{buffer.shape[-1]}] __attribute__((aligned(64))) = {{0.0}}\n")

        # forward declare gpu pid vars
        src = dec_header
        if node.parallel:
            nest_vars = [node.var] + [lv.var for lv in node.parallel_nest_loop]
            p_var = "_".join(nest_vars)
            nest_shape = [lv.end for lv in node.parallel_nest_loop]
            nest_stride = nest_shape[-1:]
            for i in range(len(nest_shape) - 2, -1, -1):
                nest_stride.insert(0, nest_stride[0] * nest_shape[i])

            src += need_indent + f"{p_var} = tl.program_id(0)\n"
            for idx, nt_var in enumerate(nest_vars):
                tdiv = f"//({nest_stride[idx]})" if idx < len(nest_stride) else ""
                tmod = f"%({nest_shape[idx-1]})" if idx > 0 else ""
                src += need_indent + f"{nt_var} = ({p_var}{tdiv}){tmod}\n"
            src += (need_indent + f"{common.SpecialVar().rbase} = tl.arange(0, RBLOCK)\n\n\n")

        else:
            if node.step == node.end:
                ##### TODO FIXME HACK
                for idx, g in enumerate(node.body):
                    if g.op_type in node_sets.ReduceNodeSetInternal():
                        g.is_final = True
                pass
            else:
                assert node.start == 0, "only support start from 0"
                src += need_indent+f"for {node.var} in range({node.start}, {node.end}, {node.step}):\n"
                indent += 4

        if isinstance(node.body, list):
            for idx, g in enumerate(node.body):
                src += g.code_gen(self, var_context, indent)
        else:
            src += node.body.code_gen(self, var_context, indent)
        return src

    def PostProcessBlock(
        self, node: IRNode, var_context: common.CodeGenContext, indent: int
    ):
        var_map = var_context.var_map
        need_indent = " " * indent
        to_be_handled_vars_map = node.body[0].var_need_post_process
        src = ""
        for s_var, v_var in to_be_handled_vars_map.items():
            # TODO determinate the type of Op, ReduceMax or ReduceMin or ReduceSum etc
            # hmin hmax hadd
            assert (s_var in var_map and s_var in node.global_connections), f"{s_var} not in var_map"
            op_type = node.global_connections[s_var].producers[0].op_type
            w_var = var_map[s_var]
            src += (need_indent + f"{v_var} = tl.where(triton_mask_1, {v_var}, 0.0)\n")
            if op_type == "ReduceSum":
                src += need_indent + f"{w_var} = tl.sum({v_var}, axis=0)\n"
            elif op_type == "ReduceMax":
                src += need_indent + f"{w_var} = tl.max({v_var}, axis=0)\n"
            else:
                raise NotImplementedError(f"not support {vec_func} yet")
            src += need_indent + f"{v_var} = {w_var}\n"
        return src

    def FunctionNode(
        self, node: FunctionNode, var_context: common.CodeGenContext, indent: int
    ):
        if not var_context:
            var_map = node.body[0].var_map

        func_input_arg = [f"e_{var_map[i.name]}" for i in node.input]
        func_input_arg = ", ".join(func_input_arg)

        func_output_arg = [f"e_{var_map[i.name]}" for i in node.output]
        func_output_arg = ", ".join(func_output_arg)

        dynamic_shape_arg = [f"{sp}" for sp in node.shape_var]
        dynamic_shape_arg = ", ".join(dynamic_shape_arg)
        if dynamic_shape_arg:
            dynamic_shape_arg += ", "

        func_signature = f"def {node.name}({func_input_arg}, {func_output_arg}, {dynamic_shape_arg} RBLOCK : tl.constexpr):\n"

        code = ""
        code += func_signature
        indent += 4
        need_indent = " " * indent

        assert node.hw_context is not None
        node.body[0].hw_context = node.hw_context

        code += node.body[0].code_gen(self, None, indent)

        # we need a wrapper to call the jit function
        if node.target != "triton_ort_training":
            code += f'''

def call{node.name}(*args):
    inputs_, outputs, shape = args
    inputs=[]
    for idx, i in enumerate(inputs_):
        inputs.append(torch.from_numpy(i).cuda(0))
    if not isinstance(outputs[0], torch.Tensor):
        for idx, o in enumerate(outputs):
            outputs[idx] = torch.from_numpy(o).cuda(0)
    n_elements_in_last_dim = outputs[0].shape[-1]
    paralleled_blocks = outputs[0].numel()//n_elements_in_last_dim
    grid = lambda meta: (paralleled_blocks* triton.cdiv(n_elements_in_last_dim, meta['RBLOCK']),)
    {node.name}[grid](*inputs, *outputs, *shape, RBLOCK=triton.next_power_of_2(n_elements_in_last_dim))

'''
        else:
        # when generate code for triton_ort_training. we are running in JIT mode, so all the shapes are known
            import torch
            final_shape = list(node.output[0].shape)
            torch_dtype = torch.from_numpy(np.zeros(1, dtype=node.output[0].dtype)).dtype
            for out in node.output:
                assert len(out.shape) == len(final_shape), "output shape dim not match"
                for idx, v in enumerate(final_shape):
                    final_shape[idx] = max(v,out.shape[idx])
            n_elements_in_last_dim = final_shape[-1]

            # allocate output tensor
            alloc_output_tensor_code = ""
            for idx, out in enumerate(node.output):
                alloc_output_tensor_code += f"    outputs.append(torch.zeros({tuple(out.shape)}, dtype={torch_dtype}, device=inputs[0].device))\n"

            code += f'''

def call{node.name}(inputs_):
    inputs=[]
    for inp in inputs_:
        inputs.append(_from_dlpack(inp))

    outputs=[]
{alloc_output_tensor_code}

    n_elements_in_last_dim = {n_elements_in_last_dim}
    paralleled_blocks = {np.prod(final_shape)//n_elements_in_last_dim}

    grid = lambda meta: (paralleled_blocks* triton.cdiv(n_elements_in_last_dim, meta['RBLOCK']),)
    {node.name}[grid](*inputs, *outputs, RBLOCK=triton.next_power_of_2(n_elements_in_last_dim))
    for idx, out in enumerate(outputs):
        outputs[idx] = to_dlpack(out)
    return outputs
'''
        return code

    def ModuleNode(self, node: IRNode, var_context: common.CodeGenContext, indent: int):
        code = """
import triton
import triton.language as tl
import torch
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack
"""

        for idx, func in enumerate(node.body):
            if not isinstance(func, MainFunctionForDebug):
                code += f"#the {idx}th function/sub_graph\n"
                code += "@triton.jit\n"
            code += func.code_gen(self, None, indent)
        return code

    def ComputeNode(
        self, node: ComputeNode, var_context: common.CodeGenContext, indent: int
    ):
        def gen_cpp_code_for_op(var_context: common.CodeGenContext):
            var_map = var_context.var_map
            vec_var_map = var_context.vectorized_var_set
            ori_named_vars_i = [var_map[i.name] for i in node.input]
            ori_named_vars_o = [var_map[i.name] for i in node.output]

            named_vars_i = ori_named_vars_i.copy()
            named_vars_o = ori_named_vars_o.copy()

            for i in range(len(named_vars_i)):
                # if named_vars_i[i] is constant scalar, just use it
                if named_vars_i[i] in var_map:
                    named_vars_i[i] = var_map[named_vars_i[i]][0]
                if str(named_vars_i[i]) in vec_var_map:
                    named_vars_i[i] = f"vec_{named_vars_i[i]}"

            raw_named_vars_1 = named_vars_i[1] if len(named_vars_i) > 1 else None
            if node.op_type == "Pow" and named_vars_i[1] == 0.5:
                node.op_type_ = "Sqrt"
            vectorized_prefix = "tl."

            assert len(named_vars_i) in [1, 2, 3]
            assert len(named_vars_o) == 1

            named_var_o = named_vars_o[0]
            src = ""
            if node.op_type == "Add":
                src += f"{named_var_o} = {named_vars_i[0]} + ({named_vars_i[1]})\n"
            elif node.op_type == "Sub":
                src += f"{named_var_o} = {named_vars_i[0]} - ({named_vars_i[1]})\n"
            elif node.op_type == "Div":
                src += f"{named_var_o} = {named_vars_i[0]} / ({named_vars_i[1]})\n"
            elif node.op_type == "Mul":
                src += f"{named_var_o} = {named_vars_i[0]} * ({named_vars_i[1]})\n"
            elif node.op_type == "Relu":
                src += f"{named_var_o} = {vectorized_prefix}maximum({named_vars_i[0]}, '0.f')\n"
            elif node.op_type == "Pow":
                # rewrite pow as mul
                if raw_named_vars_1 == 2:
                    src += f"{named_var_o} = {named_vars_i[0]} * {named_vars_i[0]}\n"
                elif raw_named_vars_1 == 3:
                    src += f"{named_var_o} = {named_vars_i[0]} * {named_vars_i[0]}* {named_vars_i[0]}\n"
                else:
                    src += f"{named_var_o} = {vectorized_prefix}libdevice.pow({named_vars_i[0]}, {named_vars_i[1]})\n"
            elif node.op_type == "Sqrt":
                src += f"{named_var_o} = {vectorized_prefix}sqrt({named_vars_i[0]})\n"
            elif node.op_type == "Rsqrt":
                if node.use_lib_device:
                    src += (
                        f"{named_var_o} = {vectorized_prefix}rsqrt({named_vars_i[0]})\n"
                    )
                else:
                    src += f"{named_var_o} = 1.f/{vectorized_prefix}sqrt({named_vars_i[0]})\n"
            elif node.op_type == "Cast":
                from_dtype = node.input[0].dtype
                to_dtype = node.output[0].dtype.type
                if to_dtype == np.bool_:
                    src += f"{named_var_o} = {named_vars_i[0]} != {_get_type(from_dtype)}(0)\n"
                elif to_dtype == np.float32:
                    src += f"{named_var_o} = ({named_vars_i[0]}).to(tl.float32)\n"
                elif to_dtype == np.float16:
                    src += f"{named_var_o} = ({named_vars_i[0]}).to(tl.float16)\n"
                else:
                    raise NotImplementedError(f"Cast to {to_dtype} is not supported")
            elif node.op_type == "Erf":
                src += f"{named_var_o} = {vectorized_prefix}libdevice.erf({named_vars_i[0]})\n"
            elif node.op_type == "Gelu":
                src += f"{named_var_o} = ({vectorized_prefix}libdevice.erf({named_vars_i[0]}/1.41421356237)+1.0)*0.5\n"
            elif node.op_type == "Exp":
                src += f"{named_var_o} = {vectorized_prefix}exp({named_vars_i[0]})\n"
            elif node.op_type == "Tanh":
                src += f"{named_var_o} = {vectorized_prefix}libdevice.tanh({named_vars_i[0]})\n"
            elif node.op_type == "Where":
                src += f"{named_var_o} = {vectorized_prefix}where({named_vars_i[0]},{named_vars_i[1]},{named_vars_i[2]})\n"
            elif node.op_type == "Sigmoid":
                if node.use_lib_device:
                    src += f"{named_var_o} = {vectorized_prefix}libdevice.sigmoid({named_vars_i[0]})\n"
                else:
                    src += f"{named_var_o} = {vectorized_prefix}sigmoid({named_vars_i[0]})\n"
            elif node.op_type == "Log":
                if node.use_lib_device:
                    src += f"{named_var_o} = {vectorized_prefix}libdevice.log({named_vars_i[0]})\n"
                else:
                    src += f"{named_var_o} = {vectorized_prefix}log({named_vars_i[0]})\n"
            else:
                raise Exception(f"not supported {node.op_type}")
            return src

        space_indent = " " * indent
        src = space_indent + f"# {node.op_name} {node.op_type}\n"
        src += space_indent + gen_cpp_code_for_op(var_context)
        return src

    def ReduceNode(self, node: ReduceNode, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        vec_var_map = var_context.vectorized_var_set
        code = "\n"
        input_key = [i.name for i in node.input]
        output_key = [i.name for i in node.output]
        out_dtype = _get_type(node.output[0].dtype)
        try:
            input_1 = var_map[var_map[input_key[1]]]
        except BaseException:
            input_1 = np.array([np.NaN])
            pass
        assert len(input_key) == 1 or (input_1[0] != np.NaN).all()
        named_var_i = var_map[input_key[0]]
        named_var_o = var_map[output_key[0]]
        # this var is vectorized, add prefix 'vec_'
        vec_pre = "tl."
        assert node.vectorization, "ReduceNode in triton must be vectorized"
        if named_var_o in vec_var_map:
            named_var_o = "vec_" + named_var_o
        if named_var_i in vec_var_map:
            named_var_i = "vec_" + named_var_i
        if named_var_i != input_key[0]:
            code += " " * indent + f"# {named_var_i} = {input_key[0]}\n"
            code += " " * indent + f"# {named_var_o} = {output_key[0]}\n"

        # FIXME, shouldn't use is_final
        if node.is_final:
            default_value = '0.0' if node.body.op_type == "ReduceSum" else 'float("-inf")'
            code += (" " * indent + f"{named_var_i} = {vec_pre}where(triton_mask_1, {named_var_i}, {default_value})\n")
            if node.body.op_type == "ReduceSum":
                code += (" " * indent + f"{named_var_o} = {vec_pre}sum({named_var_i}, 0)\n")
            elif node.body.op_type == "ReduceMax":
                code += (" " * indent + f"{named_var_o} = {vec_pre}max({named_var_i}, 0)\n")
            else:
                raise Exception(f"not supported {node.body.op_type}")
        else:
            if node.body.op_type == "ReduceSum":
                code += (" " * indent + f"{named_var_o} += {named_var_i}\n")
            elif node.body.op_type == "ReduceMax":
                code += (" " * indent +
                        f"{named_var_o} = {vec_pre}where(({named_var_o} < {named_var_i}), {named_var_i}, {named_var_o})\n")
            else:
                raise Exception(f"not supported {node.body.op_type}")
        return code

    def MaskLoadNode(self, node: MaskLoadNode, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        vec_var_map = var_context.vectorized_var_set
        space_indent = " " * indent
        code = ""
        var_name, mask_name = (i.name for i in node.input)
        input_buf:ComputeBuffer = node.input[0]
        assert var_name in var_map, f"name {var_name} not found in var_map"
        named_var = var_map[var_name]

        dtype = _get_type(input_buf.dtype)
        assert node.vectorization, "LoadNode should be vectorized in triton"

        if named_var != var_name:
            code += space_indent + f"#load  {var_name} ===>> {named_var}\n"
        annotated_var = Indexer().code_gen(named_var, input_buf, for_triton=True)

        load_addr = f"e_{annotated_var}"
        vec_var_map.add(named_var)

        rbase = var_map[common.SpecialVar().rbase]

        if input_buf.attr_cross_loop:
            return code + space_indent + f"vec_{named_var} = {load_addr}[i_0*RBLOCK:(i_0+1)*RBLOCK]\n"

        if input_buf.shape == [] or input_buf.shape[-1] == 1:
            mask_name = ''
        else:
            code += space_indent + f"roffset = {rbase} # + offset\n"
            code += space_indent + f"{mask_name} = roffset < tl.minimum(RBLOCK,{input_buf.shape[-1]})\n"
            load_addr = f"{load_addr}+roffset"
        return (code + space_indent + f"vec_{named_var} = tl.load({load_addr}, {mask_name}, other=0.0)\n")

    def MaskStoreNode(self, node: MaskStoreNode, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        code = ""
        space_indent = code + " " * indent
        var_name, mask_name = (i.name for i in node.input)
        input_buf = node.input[0]
        assert var_name in var_map
        named_var = var_map[var_name]

        if named_var != var_name:
            code += " " * indent + f"# store {var_name} <<=== {named_var}\n"
        annotated_var = Indexer().code_gen(named_var, input_buf, for_triton=True)
        assert node.vectorization, "StoreNode should be vectorized in triton"
        rbase = var_map[common.SpecialVar().rbase]

        if input_buf.attr_cross_loop:
            return code + space_indent + f"e_{annotated_var}[i_0*RBLOCK:(i_0+1)*RBLOCK] = {named_var}\n"

        code += space_indent + f"roffset = {rbase} # + offset\n"
        code += space_indent + f"{mask_name} = roffset <  tl.minimum(RBLOCK,{input_buf.shape[-1]})\n"
        if input_buf.dtype.type == np.bool_:
            assert False, "bool type not supported"
        else:
            return code + (space_indent + f"tl.store(e_{annotated_var}+roffset, {named_var}, {mask_name})\n")

    def ExecutionBlock(self, node: ExecutionBlock, var_context: common.CodeGenContext, indent: int):
        assert not var_context
        var_context = common.CodeGenContext(node.var_map)
        var_map = var_context.var_map
        need_indent = " " * indent
        src = ""

        dec_for_sub_loop = ""
        var_declared = set()
        if node.forward_var_set:
            for fvs in node.forward_var_set:
                for str_var, buffer_l in fvs.items():
                    assert len(buffer_l) == 1 if isinstance(buffer_l, list) else True
                    buffer = buffer_l[0] if isinstance(buffer_l, list) else buffer_l
                    if buffer.shape[-1] == 1:
                        continue
                    if str_var in var_declared:
                        fvs.pop(str_var)
                        continue
                    var_declared.add(str_var)
                    initialize_assign = f'= tl.zeros([tl.cdiv({buffer.shape[-1]},RBLOCK)*RBLOCK], dtype=tl.{buffer.dtype.name})'
                    dec_for_sub_loop += (need_indent + f"e_{var_map[str_var]} {initialize_assign}\n")
                    fvs.pop(str_var)
            dec_for_sub_loop += '\n'
        src += dec_for_sub_loop
        src += node.body.code_gen(self, var_context, indent)
        return src
