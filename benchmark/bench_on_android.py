import subprocess
import shutil
import onnx
from pathlib import Path
from typing import List
import re

from data.get_test_data import (
    get_backbone_onnx_path,
    get_tokenizer_and_huggingface_model,
)

def prepare_recipe(work_dir:Path, device:str, artifacts:List[Path]):
    # subprocess.check_call(args=[f"adb -s {device} shell \"rm -rf {work_dir}|| true\""], shell=True)
    subprocess.check_output(args=[f"adb -s {device} shell \"mkdir -p {work_dir}\""], shell=True)

    for artifact in artifacts:
        subprocess.check_output(
            f"adb -s {device} shell \"rm -rf {work_dir}/{artifact.name}\"", shell=True)
        subprocess.check_output(
            f"adb -s {device} push {artifact} {work_dir}/", shell=True)

def do_benchmark(output_path, bert_onnx_model, model_name, device, lib_path):
    tokenizer, hg_model, _, text = get_tokenizer_and_huggingface_model(model_name)
    encoded_input = tokenizer(*text, return_tensors="pt")

    # save inputs data for debug
    test_data_dir = output_path.parent / "test_data"
    shutil.rmtree(str(test_data_dir), ignore_errors=True)
    test_data_dir.mkdir(exist_ok=True)

    if model_name == 'microsoft/deberta-base':
        encoded_input.pop('token_type_ids')
    for idx, (k, v) in enumerate(encoded_input.items()):
        input_tensor = onnx.numpy_helper.from_array(v.numpy(), name=k)
        open(f"{test_data_dir}/input_{idx}.pb", "wb").write(
            input_tensor.SerializeToString()
        )

    base_dir = Path('/data/local/tmp')
    work_dir = base_dir/'tmp_test_aot'

    prepare_recipe(work_dir, device, 
    [lib_path, output_path,test_data_dir, bert_onnx_model, Path('/home/stcadmin/work/onnxruntime/build/Androidv8/Release/onnxruntime_perf_test')])
    # run benchmark
    cmd = f"adb -s {device} shell \"cd {work_dir} && ./onnxruntime_perf_test -m times -r 15 -x 4 {work_dir}/{output_path.name}\""
    output_aot = subprocess.check_output(cmd, shell=True).decode("utf-8")
    avg_aot = re.findall(r'Average inference time cost: ([\d.]+) ms', output_aot)[0]
    cmd = f"adb -s {device} shell \"cd {work_dir} && ./onnxruntime_perf_test -m times -r 15  -x 4 {work_dir}/{bert_onnx_model.name}\""
    output_ort = subprocess.check_output(cmd, shell=True).decode("utf-8")
    avg_ort = re.findall(r'Average inference time cost: ([\d.]+) ms', output_ort)[0]
    c_tc = [float(x) for x in [avg_ort, avg_aot]]
    print(
            f"time-cost changes from {c_tc[0]:.6}ms to {c_tc[1]:.6}ms, speedup: {(c_tc[0]/c_tc[1]):.2f}x"
        )
    return c_tc
