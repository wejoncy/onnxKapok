from collections import defaultdict,OrderedDict
import onnx
import copy
import symbolic_shape_infer


class SpecialVar(object):
    def __init__(self):
        self.input_args = "input_args"
        self.output_args = "output_args"
        self.input_args_size =  "input_args_size"
        self.parallel_loop_start = "p_loop_start"
        self.parallel_loop_end = "p_loop_end"

def add_all_intermidiate_values(model):
    model_proto = copy.deepcopy(model)
    # model_proto, check = simplify(model_proto)
    org_outputs = [x.name for x in model_proto.graph.output]
    for node in model_proto.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model_proto.graph.output.extend([onnx.ValueInfoProto(name=output)])
    return model_proto


def get_symbol_shape(model_path):
    if isinstance(model_path, str):
        model = onnx.load(model_path)
    else:
        model = model_path
    symbol_shape = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
        model, 2**31 - 1, True, guess_output_rank=True, verbose=1
    )

    return symbol_shape


class OnnxInGraph(object):
    def __init__(self, onnx_model: onnx.ModelProto):
        self.model_proto = onnx_model
        self.graph = self.model_proto.graph

        self.node_name2module = dict()
        self.produced_by = dict()
        self.consumed_by = dict()
        self.graph_input_names = []
        self.graph_output_names = []
        self.initializer_name2module = dict()
        self.tensor_type_shape_info = dict()
        self.value_info_map = dict()
        #####

    @staticmethod
    def get_all_shape_from_onnx_model(model):
        symbol_shape = get_symbol_shape(model)
        if symbol_shape is not None:
            type_shape_dict = dict()
            for value_info in symbol_shape.graph.value_info:
                dim = value_info.type.tensor_type.shape.dim
                type_shape_dict[value_info.name] = (
                    value_info.type.tensor_type.elem_type,
                    [d.dim_value or d.dim_param for d in dim],
                )

            for inp in model.graph.input:
                dim = inp.type.tensor_type.shape.dim
                assert inp.name not in type_shape_dict, (
                    "repeat input name: %s" % inp.name
                )
                type_shape_dict[inp.name] = (
                    inp.type.tensor_type.elem_type,
                    [d.dim_value or d.dim_param for d in dim],
                )

            return type_shape_dict,{}
        import transformers
        import onnxruntime as ort
        model_proto = add_all_intermidiate_values(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            #"distilbert-base-uncased"
            "lordtt13/emo-mobilebert"
        )
        inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
        #inputs["input_mask"] = inputs["attention_mask"]
        #del inputs["attention_mask"]
        sess = ort.InferenceSession(path_or_bytes=model_proto.SerializeToString())
        if sess.get_inputs()[0].name == "input_ids":
            ret = sess.run([i.name for i in sess.get_outputs()], dict(inputs))
            runtime_shape = {
                key.name: ret[idx].shape for idx, key in enumerate(sess.get_outputs())
            }
        else:
            runtime_shape={}

        return type_shape_dict,runtime_shape

    def gen_name2module_map(self):
        for vi in self.graph.value_info:
            self.value_info_map[vi.name] = vi
        # initializer name => initializer
        for initializer in self.graph.initializer:
            self.initializer_name2module[initializer.name] = initializer

        # node name => node
        node_idx = 0
        for node in self.graph.node:
            if node.name == "":
                node.name = str(node.op_type) + str(node_idx)
            
            node_idx += 1
            self.node_name2module[node.name] = node
            for out in node.output:
                # if node.op_type == "Constant":
                #    continue
                assert out not in self.produced_by, (
                    "multiple producers for node: %s" % out
                )
                self.produced_by[out] = [node]
            for inp in node.input:
                if inp in self.initializer_name2module:
                    continue
                if inp not in self.consumed_by:
                    self.consumed_by[inp] = []
                self.consumed_by[inp].append(node)

        for inp in self.graph.input:
            self.node_name2module[inp.name] = inp
        self.graph_input_names.extend([inp.name for inp in self.graph.input])

        for out in self.graph.output:
            self.node_name2module[
                "out_" + out.name
            ] = out  # add `out_` in case the output has the same name with the last node
        self.graph_output_names = ["out_" + out.name for out in self.graph.output]
        symbol_shape,rt_shape = self.get_all_shape_from_onnx_model(self.model_proto)
        self.tensor_type_shape_info.update(
            symbol_shape
        )
        self.rt_shape= rt_shape

class GraphIOBuffer(object):
    def __init__(self):
        self.var_buffer_in = []
        self.var_buffer_out = []
        self.const_buffer = []


class IndexSubGraph(object):
    def __init__(self):
        self.sub_graph_nodes = []
        self.input_name_exclude_constant = OrderedDict()
        self.input_name_ref_c = OrderedDict()
        self.output_name_ref_c = OrderedDict()
        self.reduce_op = []

    def analyze_input_output(self, consumed_by_name2node):
        for ink in list(self.input_name_ref_c.keys()):
            v = self.input_name_ref_c[ink]
            if ink in self.output_name_ref_c:
                self.input_name_ref_c.pop(ink)
                if len(consumed_by_name2node[ink]) == v:
                    self.output_name_ref_c.pop(ink)
