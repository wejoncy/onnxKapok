import sys

sys.path.insert(0, r"/home/stcadmin/work/onnxruntime/build/py38/Release/build/lib")
import onnxruntime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from data.get_test_data import (
    get_backbone_onnx_path,
    get_tokenizer_and_huggingface_model,
)

import transformers
import numpy as np

import ort_aot


def verify_results(model_file: Path, model_name: str, onnx_bert_model: Path = None):
    """
    Args:
        model_file: the onnx model which finalized and needs to be verified
        model_name: the huggingface model name
        onnx_bert_model: the onnx model which is generated by huggingface or user provide
    """
    tokenizer, hg_model, _, text = get_tokenizer_and_huggingface_model(model_name)
    encoded_input = tokenizer(*text, return_tensors="pt")
    transformers.set_seed(42)

    session_options = onnxruntime.SessionOptions()

    if onnx_bert_model.exists():
        session = onnxruntime.InferenceSession(
            str(onnx_bert_model.resolve(strict=True))
        )
        inputs = {key: value.detach().numpy() for key, value in encoded_input.items()}

        ref_outputs = session.run([i.name for i in session.get_outputs()], inputs)
        ref_map_out = {
            i.name: ref_outputs[idx] for idx, i in enumerate(session.get_outputs())
        }
    else:
        outs = hg_model(**encoded_input)
        ref_outputs = [out.detach().numpy() for out in list(outs.values())]
        ref_map_out = {i: ref_outputs[idx] for idx, i in enumerate(outs.keys())}

    session = onnxruntime.InferenceSession(
        str(model_file.resolve(strict=True)), session_options
    )

    real_outputs = session.run([i.name for i in session.get_outputs()], inputs)
    matched_idx = [
        i
        for i, o in enumerate(session.get_outputs())
        if list(ref_map_out.keys())[0] in o.name
    ][0]

    assert np.allclose(
        real_outputs[matched_idx],
        ref_outputs[0],
        atol=1e-12,
        rtol=1e-15,
    ), f"Results do not match, expected:{ref_outputs[0]}, but got {real_outputs[matched_idx] }"
    print(
        "Results matches:",
        real_outputs[0],
        "\ndiff:",
        real_outputs[matched_idx] - ref_outputs[0],
    )


def do_bench(output_model_path: Path, input_model_path: Path, model_name: str):
    tokenizer, hg_model, _, text = get_tokenizer_and_huggingface_model(model_name)
    encoded_input = tokenizer(*text, return_tensors="np")

    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 4
    session_in = onnxruntime.InferenceSession(
        str(input_model_path.resolve(strict=True)),
        session_options,
        providers=["CPUExecutionProvider"],
    )
    session_out = onnxruntime.InferenceSession(
        str(output_model_path.resolve(strict=True)),
        session_options,
        providers=["CPUExecutionProvider"],
    )
    inputs = dict(encoded_input)

    c_tc = []
    # warmup
    for _ in range(5):
        _ = session_in.run(None, inputs)
        _ = session_out.run(None, inputs)

    repeat = 1000
    with ort_aot.CostTime(c_tc, repeat) as tc:
        for i in range(repeat):
            _ = session_in.run(None, inputs)
    with ort_aot.CostTime(c_tc, repeat) as tc:
        for i in range(repeat):
            _ = session_out.run(None, inputs)

    print(
        f"the original model cost time: {c_tc[0]}ms, the aot model cost time: {c_tc[1]}ms"
    )


def run():
    model_name = "google/mobilebert-uncased"
    model_name = "xlm-roberta-base"
    model_name = "lordtt13/emo-mobilebert"
    model_name = "distilbert-base-uncased"
    bert_onnx_model = get_backbone_onnx_path(model_name)
    output_path = Path(str(bert_onnx_model).replace(".onnx", "_aot.onnx"))
    lib_path = Path(__file__).parent.resolve(strict=True) / "libcode.so"
    if output_path.exists() and lib_path.exists():
        print("bypass compiling, use cached model")
    else:
        ort_aot.compile_model(bert_onnx_model, output_path, lib_path)
    # verify_results(output_path, model_name, bert_onnx_model)
    do_bench(output_path, bert_onnx_model, model_name)


if __name__ == "__main__":
    run()
