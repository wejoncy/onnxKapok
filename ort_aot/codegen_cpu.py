from ir import *
import common


def _get_type(t):
    out_dtype = common.NP_TYPE_C_TYPE[t.type]
    return out_dtype


class CPUCodeGen(common.NodeVisitor):
    def __init__(self):
        super().__init__()

    def visit(self, node: IRNode, context: common.CodeGenContext, indent: int):
        fn = getattr(self, node.__class__.__name__)
        assert fn is not None,  "unimplemented node: %s" % node.__class__.__name__
        return fn(node, context, indent)

    def Loop(self, node: IRNode, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        vec_var_set = var_context.vectorized_var_set
        need_indent = " " * indent
        dec_header = ""
        # forward declaration
        for fvar, buffer_l in node.forward_var_set.items():
            buffer = buffer_l[0] if isinstance(buffer_l, list) else buffer_l
            str_var = str(fvar)
            if buffer.shape is not None and buffer.shape[-1] == 1:
                if str_var not in node.reduction_var:
                    init_val = 0.0
                elif 'sum' in node.reduction_var[str_var].lower():
                    init_val = 0.0
                elif 'max' in node.reduction_var[str_var].lower():
                    init_val = '-3.40082e38'
                elif 'min' in node.reduction_var[str_var].lower():
                    init_val = '3.40082e38'
                else:
                    assert False, "unsupported reduction type: %s" % node.reduction_var[str_var]

                dec_header += need_indent + f"float {var_map[str_var]} = {init_val}f;\n"
                if node.vectorization:
                    dtype = _get_type(buffer.dtype)
                    dec_header += (need_indent + f"mipp::Reg<{dtype}> vec_{var_map[str_var]} = 0.0f;\n")
                    vec_var_set.add(var_map[str_var])
                    node.var_need_post_process[str_var] = f"vec_{var_map[str_var]}"
            else:
                assert False, "buffer should be defined in the execution-block"
                dec_header += (
                    need_indent
                    + f"float e_{var_map[str_var]}[{buffer.shape[-1]}] __attribute__((aligned(64))) = {{0.0}};\n"
                )

        # forward declare vectorization vars
        v_step = ""
        if node.vectorization:
            v_step = "mipp::N<float>()"
        src = dec_header
        src += need_indent + f"//@{node.attributes.name}\n"
        src += need_indent
        if node.parallel:
            p_var = f"{node.var}"
            p_var += (f"_{node.parallel_nest_loop.var}" if node.parallel_nest_loop else "")
            src += f"for (int {p_var}={common.SpecialVar().parallel_loop_start}; {p_var}<{common.SpecialVar().parallel_loop_end}; {p_var}+={node.step}){{\n"
            if node.parallel_nest_loop:
                src += (need_indent+need_indent + f"auto {node.var} = {p_var}/{node.parallel_nest_loop.end};\n")
                nest_var = node.parallel_nest_loop.var
                src += (need_indent+need_indent + f"auto {nest_var} = {p_var}%{node.parallel_nest_loop.end};\n")
        else:
            # if node.depth == 0 and node.start == 0:
            #    if not node.end.is_Number:
            #        src += f'#pragma omp assume holds({node.end} % 16 == 0 && {node.end} > 0)\n'+need_indent
            #    src += f'#pragma omp simd\n'+need_indent
            #    src += f'#pragma vector aligned\n'+need_indent
            src += f"for (int {node.var}={node.visit(node.start)}; {node.var}<{node.visit(node.end)}; \
{node.var}+={v_step if node.vectorization else node.step}){{\n"

        if isinstance(node.body, list):
            for idx, g in enumerate(node.body):
                src += g.code_gen(self, var_context, indent + 4)
        else:
            src += node.body.code_gen(self, var_context, indent + 4)
        src = src + need_indent + "}\n"
        return src

    def PostProcessBlock(self, node: IRNode, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        need_indent = " " * indent
        to_be_handled_vars_map = node.body[0].var_need_post_process
        src = ""
        for s_var, v_var in to_be_handled_vars_map.items():
            # TODO determinate the type of Op, ReduceMax or ReduceMin or ReduceSum etc
            # hmin hmax hadd
            assert (
                s_var in var_map and s_var in node.global_connections
            ), f"{s_var} not in var_map"
            vec_func = node.op_map[node.global_connections[s_var].producers[0].op_type]
            w_var = var_map[s_var]
            if vec_func == "hadd":
                src += need_indent + f"{w_var} = {w_var} + mipp::{vec_func}({v_var});\n"
            elif vec_func == "hmax":
                src += (need_indent + f"{w_var} = std::max({w_var} , mipp::{vec_func}({v_var}));\n")
            else:
                raise NotImplementedError(f"not support {vec_func} yet")
            src += need_indent + f"{v_var} = {w_var};\n"
        return src

    def FunctionNode(self, node: IRNode, var_context: common.CodeGenContext, indent: int):
        if not var_context:
            var_map = node.body[0].var_map
        # node.output[0].dtype.tensor_type.shape.dim[1]

        # in_param = [f"const float* e_{var_map[i.name]}" for i in node.input]

        in_param = [f"const void** {common.SpecialVar().input_args}",]
        if node.body[0].body.parallel:
            in_param += [
                f" ptrdiff_t  {common.SpecialVar().parallel_loop_start}, ptrdiff_t {common.SpecialVar().parallel_loop_end}"
            ]

        in_param += [f"const int64_t* {common.SpecialVar().dynamic_shape_args}"]
        in_param = ",".join(in_param)

        out_dtype = [_get_type(out.dtype) for out in node.output]
        # out_param = ",".join([f"float* e_{var_map[i.name]}" for i in node.output])
        out_param = f"void** output_args"

        func_signature = f"int {node.name}({in_param}, {out_param}) {{\n"

        code = ""
        code += func_signature
        indent += 4
        need_indent = " " * indent
        # DEBUG code

        assert node.hw_context is not None
        bytes_lanes = node.hw_context.vec_lanes * 4  # (sizeof(float)

        in_dtype = [_get_type(i.dtype) for i in node.input]
        restrict = f"__restrict__  __attribute__((aligned ({bytes_lanes})))"
        parse_input = [
            need_indent
            + f"const {in_dtype[idx]}* {restrict} e_{var_map[i.name]} = (const {in_dtype[idx]}*){common.SpecialVar().input_args}[{idx}];"
            for idx, i in enumerate(node.input)
        ]
        code += "\n".join(parse_input) + "\n\n"

        parse_shape = [
            need_indent
            + f"const int64_t {sp}  = {common.SpecialVar().dynamic_shape_args}[{idx}];"
            for idx, sp in enumerate(node.shape_var)
        ]
        code += "\n".join(parse_shape) + "\n\n"

        parse_output = [
            need_indent
            + f"{out_dtype[idx]}* {restrict} e_{var_map[i.name]} = ({out_dtype[idx]}*){common.SpecialVar().output_args}[{idx}];"
            for idx, i in enumerate(node.output)
        ]
        code += "\n".join(parse_output) + "\n\n"

        de_composed_const_var = {}
        for const in node.const_var:
            if isinstance(const, onnx.NodeProto):
                if const.attribute[0].type == common.AttributeType.TENSOR:
                    assert len(const.attribute) == 1
                    de_composed_const_var[const.output[0]] = const.attribute[0].t
                else:
                    de_composed_const_var[const.output[0]] = const
            else:
                de_composed_const_var[const.name] = const

        for name, const in de_composed_const_var.items():
            np_array = common.parse_onnx_to_numpyarray(const)
            dtype = _get_type(np_array.dtype)
            if np_array.size > 1:
                np_array_u32 = np_array.view(np.uint32)
                x_arrstr = np.char.mod("%#x", np_array_u32)
                x_str = ",".join(x_arrstr)
                dtype = _get_type(np_array.dtype)
                const_declare = (
                    f"static constexpr uint32_t e_{var_map[name]}_u32p[] __attribute__((aligned({bytes_lanes}))) = {{{x_str}}};\n"
                    + need_indent
                    + f"static const {dtype}* e_{var_map[name]} = (const {dtype}*)e_{var_map[name]}_u32p;\n"
                )
            else:
                # we have expanded the const var to scalar
                continue
                # suffix = "f" if np_array.dtype == np.float32 else ""
                # v = np_array.reshape(-1)[0]
                # const_declare = f"static constexpr {dtype} e_{var_map[name]} = {v}{suffix};\n"
            code += need_indent + const_declare

        node.body[0].hw_context = node.hw_context

        code += node.body[0].code_gen(self, None, indent)
        code += need_indent + "return 12;\n"
        code += "}\n"

        return code

    def ModuleNode(self, node: IRNode, var_context: common.CodeGenContext, indent: int):
        code = """
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstddef>
"""
        if node.has_vectorization:
            code += """
#include <mipp/mipp.h>
#define __SIMD__ 1
"""
        code += """
#include <fast_math.h>
"""
        code += """
extern "C"{
"""

        for idx, func in enumerate(node.body):
            code += f"//the {idx}th function/sub_graph\n"
            code += func.code_gen(self, None, indent)
        # extern C
        code += "}\n"
        return code

    def ComputeNode(self, node: ComputeNode, var_context: common.CodeGenContext, indent: int):
        def gen_cpp_code_for_op(var_context: common.CodeGenContext):
            var_map = var_context.var_map
            vec_var_map = var_context.vectorized_var_set
            ori_named_vars_i = [var_map[i.name] for i in node.input]
            ori_named_vars_o = [var_map[i.name] for i in node.output]
            suffix = ["" for i in ori_named_vars_i]

            named_vars_i = ori_named_vars_i.copy()
            named_vars_o = ori_named_vars_o.copy()

            for i in range(len(named_vars_i)):
                # if named_vars_i[i] is constant scalar, just use it
                if named_vars_i[i] in var_map:
                    named_vars_i[i] = var_map[named_vars_i[i]][0]
                    if node.input[i].dtype == np.float32:
                        # without suffix 'f', compiler will see it as double
                        suffix[i] = "f"

            vectorized_prefix = "std::"
            # use it to check if the input is a vector
            # for vectorized op, we need to add "vec_" for reduced output vars
            # in tile_loop, we split loops into main loop and tail loop
            # main loop is vectorized, tail loop is not.
            # so we have two kinds of output vars
            is_input_1_vec = False
            if node.vectorization:
                vectorized_prefix = "mipp::"
                for idx, var in enumerate(named_vars_i):
                    if str(var) in vec_var_map:
                        named_vars_i[idx] = f"vec_{var}"
                        is_input_1_vec = idx == 1
            # canonize sub as add
            if node.op_type == "Sub" and (
                not isinstance(named_vars_i[0], str) or is_input_1_vec
            ):
                named_vars_i[1] = "-" + named_vars_i[1]
                node.op_type_ = "Add"

            if (
                node.op_type == "Div"
                and is_input_1_vec
                and not named_vars_i[0].startswith("vec_")
            ):
                dtype = _get_type(node.input[0].dtype)
                named_vars_i[0] = f"mipp::Reg<{dtype}>({named_vars_i[0]})"

            if (
                node.op_type == "Pow" and named_vars_i[1] == 0.5
            ):
                node.op_type_ = "Sqrt"

            # always trying to put the constant to the right
            if (not isinstance(named_vars_i[0], str) or is_input_1_vec) and node.op_type in ["Add", "Mul"]:
                suffix[0], suffix[1] = suffix[1], suffix[0]
                named_vars_i[0], named_vars_i[1] = named_vars_i[1], named_vars_i[0]

            raw_named_vars_1 = named_vars_i[-1]

            # add suffix 'f' for float constant
            for idx, var in enumerate(named_vars_i):
                named_vars_i[idx] = f"{var}{suffix[idx]}"

            if len(named_vars_i) == 3 and suffix[-1] == "f" and node.vectorization:
                dtype = _get_type(node.input[-1].dtype)
                named_vars_i[-1] = f"mipp::Reg<{dtype}>({named_vars_i[-1]})"

            assert len(named_vars_i) in [1, 2, 3]
            assert len(named_vars_o) == 1

            named_var_o = named_vars_o[0]
            src = "auto "
            if node.op_type == "Add":
                src += f"{named_var_o} = {named_vars_i[0]} + ({named_vars_i[1]});\n"
            elif node.op_type == "Sub":
                src += f"{named_var_o} = {named_vars_i[0]} - ({named_vars_i[1]});\n"
            elif node.op_type == "Div":
                src += f"{named_var_o} = {named_vars_i[0]} / ({named_vars_i[1]});\n"
            elif node.op_type == "Mul":
                src += f"{named_var_o} = {named_vars_i[0]} * ({named_vars_i[1]});\n"
            elif node.op_type == "Relu":
                src += f"{named_var_o} = {vectorized_prefix}max<float>({named_vars_i[0]}, {'vec_zero' if node.vectorization else '0.f'});\n"
            elif node.op_type == "Fma":
                # a*b+c
                src += f"{named_var_o} = std::fma({named_vars_i[0]}, ({named_vars_i[1]},{named_vars_i[2]}));\n"
            elif node.op_type == "Pow":
                # rewrite pow as mul
                if raw_named_vars_1 == 2:
                    src += f"{named_var_o} = {named_vars_i[0]} * {named_vars_i[0]};\n"
                elif raw_named_vars_1 == 3:
                    src += f"{named_var_o} = {named_vars_i[0]} * {named_vars_i[0]}* {named_vars_i[0]};\n"
                else:
                    src += f"{named_var_o} = pow({named_vars_i[0]},{named_vars_i[1]});\n"
            elif node.op_type == "Sqrt":
                src += f"{named_var_o} = {vectorized_prefix}sqrt({named_vars_i[0]});\n"
            elif node.op_type == "Rsqrt":
                if node.vectorization:
                    src += f"{named_var_o} = {vectorized_prefix}rsqrt({named_vars_i[0]});\n"
                else:
                    src += f"{named_var_o} = 1.f/{vectorized_prefix}sqrt({named_vars_i[0]});\n"
            elif node.op_type == "Cast":
                from_dtype = node.input[0].dtype
                to_dtype = node.output[0].dtype.type
                if to_dtype == np.bool_:
                    src += f"{named_var_o} = {named_vars_i[0]} != {_get_type(from_dtype)}(0);\n"
                else:
                    src += f"{named_var_o} = ({named_vars_i[0]});\n"
            elif node.op_type == "Erf":
                # 2/sqrt(M_PI) = 1.1283791671f
                src += f"{named_var_o} = tanh_mlas({named_vars_i[0]}*({named_vars_i[0]}*{named_vars_i[0]}*2*0.044715f+1.f)*1.1283791671f);\n"
            elif node.op_type == "Gelu":
                # sqrt(2/M_PI) = 0.7978845608f
                src += f"{named_var_o} = {named_vars_i[0]}*(tanh_mlas( ({named_vars_i[0]}*({named_vars_i[0]}* {named_vars_i[0]}*0.044715f+1.0f) *0.7978845608f ))+1.0f)*0.5f;\n"
            elif node.op_type == "Exp":
                src += f"{named_var_o} = {vectorized_prefix}exp({named_vars_i[0]});\n"
            elif node.op_type == "Tanh":
                src += f"{named_var_o} = tanh_mlas({named_vars_i[0]});\n"
            elif node.op_type == "Where":
                if node.vectorization:
                    src += f"{named_var_o} = {vectorized_prefix}blend({named_vars_i[1]},{named_vars_i[2]},{named_vars_i[0]});\n"
                else:
                    src += f"{named_var_o} = {named_vars_i[0]} ? {named_vars_i[1]} : {named_vars_i[2]};\n"
            else:
                raise Exception(f"not supported {node.op_type}")
            return src

        space_indent = " " * indent
        src = space_indent + f"// {node.op_name} {node.op_type}\n"

        if node.op_type == "Relu" and node.vectorization:
            src += space_indent + "mipp::Reg<float> vec_zero = 0.0f;\n"
        src += space_indent + gen_cpp_code_for_op(var_context)
        return src

    def ReduceNode(self, node: IRNode, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        vec_var_map = var_context.vectorized_var_set
        code = "\n"
        input_key = [i.name for i in node.input]
        output_key = [i.name for i in node.output]
        out_dtype = _get_type(node.output[0].dtype)
        try:
            input_1 = var_map[var_map[input_key[1]]]
        except:
            input_1 = np.array([np.NaN])
            pass
        assert len(input_key) == 1 or (input_1[0] != np.NaN).all()
        named_var_i = var_map[input_key[0]]
        named_var_o = var_map[output_key[0]]
        # this var is vectorized, add prefix 'vec_'
        vec_pre = "mipp::" if node.vectorization else "std::"
        if node.vectorization:
            if named_var_o in vec_var_map:
                named_var_o = "vec_" + named_var_o
            if named_var_i in vec_var_map:
                named_var_i = "vec_" + named_var_i
        if named_var_i != input_key[0]:
            code += " " * indent + f"// {named_var_i} = {input_key[0]};\n"
            code += " " * indent + f"// {named_var_o} = {output_key[0]};\n"
        if node.body.op_type == "ReduceSum":
            code += " " * indent + \
                f"{named_var_o} = {named_var_o}+{named_var_i};\n"
        elif node.body.op_type == "ReduceMax":
            code += (
                " " * indent
                + f"{named_var_o} = {vec_pre}max<{out_dtype}>({named_var_o},{named_var_i});\n"
            )
        else:
            raise Exception(f"not supported {node.body.op_type}")
        return code

    def LoadNode(self, node: IRNode, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        vec_var_map = var_context.vectorized_var_set
        space_indent = " " * indent
        code = ""
        var_name = node.input.name
        assert var_name in var_map, f"name {var_name} not found in var_map"
        named_var = var_map[var_name]

        dtype = _get_type(node.input.dtype)
        # if named_var is constant, no need to load
        if named_var in var_map:
            assert False, f"TODO: {named_var} is constant, no need to load"
            v = var_map[named_var]
            assert isinstance(v, (np.ndarray))
            if node.vectorization:
                vec_var_map.add(named_var)
                return (code + space_indent + f"mipp::Reg<float> vec_{named_var} = {v[0]}f;\n")
            else:
                return code + space_indent + f"auto {named_var} = {v[0]}f;\n"

        if named_var != var_name:
            code += space_indent + f"//load  {var_name} ===>> {named_var};\n"
        annotated_var = Indexer().code_gen(named_var, node.input)

        load_addr = f'e_{annotated_var}'
        if node.vectorization:
            vec_var_map.add(named_var)
            if node.input.shape == [] or node.input.shape[-1] == 1:
                pass
            else:
                load_addr = f'&e_{annotated_var}'

            if node.input.dtype.type == np.bool_:
                vec_type = "mipp::Msk<mipp::N<float>()>"
            else:
                vec_type = f"mipp::Reg<{dtype}>"
            return (
                code
                + space_indent
                + f"{vec_type} vec_{named_var} = {load_addr};\n"
            )
        else:
            return code + space_indent + f"auto {named_var} = {load_addr};\n"

    def StoreNode(self, node: IRNode, var_context: common.CodeGenContext, indent: int):
        var_map = var_context.var_map
        code = ""
        space_indent = code + " " * indent
        var_name = node.to_var.name
        assert var_name in var_map
        named_var = var_map[var_name]

        if named_var != var_name:
            code += " " * indent + f"// store {var_name} <<=== {named_var};\n"
        annotated_var = Indexer().code_gen(named_var, node.to_var)
        if node.vectorization:
            # TODO special case for MIPP
            if node.to_var.dtype.type == np.bool_:
                return space_indent + f"mipp::toReg<int8_t>({named_var}).store((int8_t*)&e_{annotated_var});\n"
            else:
                return space_indent + f"{named_var}.store(&e_{annotated_var});\n"
        else:
            return space_indent + f"e_{annotated_var} = {named_var};\n"

    def ExecutionBlock(self, node: ExecutionBlock, var_context: common.CodeGenContext, indent: int):
        assert not var_context
        var_context = common.CodeGenContext(node.var_map)
        var_map = var_context.var_map
        need_indent = " " * indent
        src = ""
        # forward declaration of intermediate buffer for sub-loop for better reuse
        bytes_lanes = node.hw_context.vec_lanes * 4
        dec_for_sub_loop = ""
        var_declared = set()
        if node.forward_var_set:
            for fvs in node.forward_var_set:
                for str_var, buffer_l in tuple(fvs.items()):
                    assert len(buffer_l) == 1 if isinstance(
                        buffer_l, list) else True
                    buffer = buffer_l[0] if isinstance(
                        buffer_l, list) else buffer_l
                    if buffer.shape[-1] == 1:
                        continue
                    if str_var in var_declared:
                        fvs.pop(str_var)
                        continue
                    var_declared.add(str_var)
                    initialize_assign = f'= {{0.0}}' if buffer.shape[-1].is_number else ''
                    dec_for_sub_loop += (
                        need_indent
                        + f"float e_{var_map[str_var]}[{buffer.shape[-1]}] __attribute__((aligned({bytes_lanes}))) {initialize_assign};\n"
                    )
                    fvs.pop(str_var)
        src += dec_for_sub_loop
        src += node.body.code_gen(self, var_context, indent)
        return src


class MainFunctionForDebug(IRNode):
    def __init__(self, function: FunctionNode):
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
        w_vars = [f'output{i}' for i in range(len(output_shapes))]
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
