import onnx
import onnx.numpy_helper


class ElementWiseNodeSet:
    type_collection = set(
        {
            "Add",
            "Sub",
            "Pow",
            "Sqrt",
            "Div",
            "Mul",
            "Exp",
            "Tanh",
            "Cast",
            "Erf",
            "Gelu",
            "FastGelu",
            "Relu",
            #"Equal",
            #"Not",
            #"Where",
        }
    )

    def __contains__(self, optype: str):
        return optype in self.type_collection


class ElementMoveNodeSet:
    type_collection = set(
        {
            "Gather",
            # "Unsqueeze",
            "Slice",
            "Concat",
            # "Reshape",
            # "Expand",
            "transpose",
        }
    )

    def __contains__(self, optype):
        return optype in self.type_collection


class ShapeNodeSet:
    type_collection = set(
        {
            "Unsqueeze",
            "Slice",
            "Concat",
            "Reshape",
            "Expand",
        }
    )

    def __contains__(self, optype):
        return optype in self.type_collection

class DecomposeNodeSetInternal:
    type_collection = set(
        {
            "ReduceMean",
            "Softmax",
            "LayerNormalization",
        }
    )

    def __contains__(self, node_or_optype):
        if isinstance(node_or_optype, onnx.NodeProto):
            optype = node_or_optype.op_type
        else:
            optype = node_or_optype
        return optype in self.type_collection
class ReduceNodeSetInternal:
    type_collection = set(
        {
            "ReduceMean",
            "ReduceSum",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceSumSquare",
            "ReduceL1",
            "ReduceL2",
            "Sum",
            "Max",
            "Min",
            "Prod",
            "LogSum",
            "LogSumExp",
            "SumSquare",
            "L1",
            "L2",
            "Softmax",
            "LayerNormalization",
        }
    )

    def __contains__(self, node_or_optype):
        try:
            optype = node_or_optype.op_type
        except:
            optype = node_or_optype
        return optype in self.type_collection


class ReduceNodeSet:
    def __init__(self, produce_by: dict):
        self.produce_by = produce_by
        self.type_collection = ReduceNodeSetInternal().type_collection

    def is_support_Softmax(self, node: onnx.NodeProto):
        if node.attribute[0].name != "axis" or node.attribute[0].i != -1:
            return False
        return True

    def __contains__(self, node: onnx.NodeProto):
        if not isinstance(node, onnx.NodeProto):
            if node is not None and not isinstance(node, onnx.ValueInfoProto):
                raise TypeError("node should be an onnx.NodeProto")
            return False

        if node.op_type == "Softmax":
            return self.is_support_Softmax(node)
        if not node or node.op_type not in self.type_collection:
            return False
        if len(node.input) > 1:
            assert (
                len(node.input) == 2
            ), "Reduce node should have only one or two inputs"
            if self.produce_by[node.input[1]][0].op_type != "Constant":
                return False
            axes_tensor = self.produce_by[node.input[1]][0].attribute[0].t
            numpy_axes = onnx.numpy_helper.to_array(axes_tensor)
            # only support the last axis
            if numpy_axes.size != 1 or numpy_axes[0] != -1:
                return False
        attrs = node.attribute
        for attr in attrs:
            if attr.name == "axes":
                if len(attr.ints) != 1 or attr.ints[0] != -1:
                    return False
            elif attr.name == "keepdims":
                if attr.i != 1:
                    return False

        return True

    def support(self, optype):
        return optype in self.type_collection
