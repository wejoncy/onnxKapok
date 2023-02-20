import onnx
import numpy as np


class DecomposeDispatch(object):
    def __init__(self):
        super().__init__()
        self.count = 0

    def get_unique_var_name(self, prefix):
        self.count += 1
        return prefix + str(self.count)

    def new_node(self, op_type, inputs, name, output=None, **kwargs):
        name = self.get_unique_var_name(name)
        if output is None:
            output = [self.get_unique_var_name(f"{op_type}_out")]
        return output[0], onnx.helper.make_node(op_type, inputs, output, name, **kwargs)

    def __call__(self, node:onnx.NodeProto, *args, **kwargs):
        if not hasattr(self, node.op_type):
            raise NotImplementedError(
                "Not implemented for op type: {}".format(node.op_type)
            )
        return getattr(self, node.op_type)(node, *args, **kwargs)

    def LayerNormalization(self, node: onnx.NodeProto, **kwargs):
        inputs = node.input
        outputs = node.output
        assert len(outputs)==1, "LayerNormalization should have only one output"
        axes_v = onnx.helper.make_tensor(
            name="axes",
            data_type=onnx.TensorProto.INT64,
            dims=(1,),
            vals=np.array([-1]),
        )
        e1_12_v = onnx.helper.make_tensor(
            name="f",
            data_type=onnx.TensorProto.FLOAT,
            dims=(1,),
            vals=np.array([1e-12]),
        )
        axes_out, axes_node= self.new_node("Constant", [], f"axes_in_ln_{node.name}", value=axes_v)
        e1_12_out, e1_12_node = self.new_node(
            "Constant", [], f"e1_12_{node.name}", value=e1_12_v)
        reducemean_out, reducemean_node = self.new_node(
            "ReduceMean", [inputs[0]], f"{node.name}_reducemean", axes=[-1])
        sub_out, sub_node = self.new_node("Sub", [inputs[0], reducemean_out], f"{node.name}_sub")
        mul_out0, mul_node0 = self.new_node(
            "Mul", [sub_out, sub_out], f"{node.name}_exp")
        reducemean_out1, reducemean_node1 = self.new_node(
            "ReduceMean", [mul_out0], f"{node.name}_reducemean", axes=[-1])
        add_out, add_node = self.new_node("Add", [reducemean_out1, e1_12_out], f"{node.name}_add")
        rsqrt_out, rsqrt_node = self.new_node("Rsqrt", [add_out], f"{node.name}_rsqrt")
        mul_out, mul_node = self.new_node("Mul", [sub_out,rsqrt_out], f"{node.name}_mul")
        mul_out1, mul_node1 = self.new_node("Mul", [inputs[1],mul_out], f"{node.name}_mul")
        add_out1, add_node1 = self.new_node(
            "Add", [inputs[2], mul_out1], f"{node.name}_add", output=outputs)
        
        return [e1_12_node, reducemean_node, sub_node, mul_node0, reducemean_node1, add_node, rsqrt_node, mul_node, mul_node1, add_node1]

    def Softmax(self, node:onnx.NodeProto, **kwargs):
        axis = node.attribute[0].i
        name = node.name
        axes_out = self.get_unique_var_name("axes_in_softmax")
        axes_v = onnx.helper.make_tensor(
            name="axes",
            data_type=onnx.TensorProto.INT64,
            dims=(1,),
            vals=np.array([axis]),
        )
        axes_node = onnx.helper.make_node(
            "Constant",
            [],
            [axes_out],
            f"axes_in_reduceMean_{node.name}",
            value=axes_v,
        )
        
        max_out = self.get_unique_var_name("sft_max_out")
        max_node = onnx.helper.make_node(
            "ReduceMax",
            [node.input[0]],#axes_out
            [max_out],
            f"{name}_max",
            axis=axis,            
        )
        sub_out = self.get_unique_var_name("sft_sub_out")
        sub_node = onnx.helper.make_node(
            "Sub",
            [node.input[0], max_out],
            [sub_out],
            f"{name}_sub",
        )
        exp_out = self.get_unique_var_name("sft_exp_out")
        exp_node = onnx.helper.make_node(
            "Exp",
            [sub_out],
            [exp_out],
            f"{name}_exp",
        )
        sum_out = self.get_unique_var_name("sft_sum_out")
        sum_node = onnx.helper.make_node(
            "ReduceSum",
            [exp_out, axes_out],
            [sum_out],
            f"{name}_sum",
        )

        div_node = onnx.helper.make_node(
            "Div",
            [exp_out, sum_out],
            node.output,
            f"sum/decomposed_from_{node.name}",
        )
        return [axes_node, max_node, sub_node, exp_node, sum_node, div_node]

    def ReduceMean(self, node:onnx.NodeProto, **kwargs):        
        axes_out = self.get_unique_var_name("axes_in_reduceMean")
        axes_v = onnx.helper.make_tensor(
            name="axes",
            data_type=onnx.TensorProto.INT64,
            dims=(1,),
            vals=np.array([node.attribute[0].ints[0]]),
        )
        axes_node = onnx.helper.make_node(
            "Constant",
            [],
            [axes_out],
            f"axes_in_reduceMean_{node.name}",
            value=axes_v,
        )
        sum_out = self.get_unique_var_name("sum_out")
        sum_node = onnx.helper.make_node(
            "ReduceSum",
            [node.input[0], axes_out],
            [sum_out],
            f"sum/decomposed_from_{node.name}",
        )

        v = onnx.helper.make_tensor(
            name="last_dim",
            dims=(),
            data_type=onnx.TensorProto.FLOAT,
            vals=np.array([768.0]),
        )
        const_v_neg1 = self.get_unique_var_name("constant_shape_in_axis")
        n_elem = onnx.helper.make_node(
            "Constant",
            [],
            [const_v_neg1],
            name=f"shape[-1]/decomposed_from_{node.name}",
            value=v,
        )
        div_node = onnx.helper.make_node(
            "Div",
            [sum_out, const_v_neg1],
            node.output,
            f"Div/decomposed_from_{node.name}",
        )
        return [axes_node, sum_node, n_elem, div_node]
