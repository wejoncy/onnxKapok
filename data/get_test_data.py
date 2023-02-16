import argparse
import os
import shutil
import re
import tempfile
import functools

from pathlib import Path

import transformers
import torch

import sys

import onnxruntime
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

def get_this_dir():
    return Path(__file__).parent.resolve()


def get_canonized_name(model_name: str):
    return re.sub(r"[^a-zA-Z0-9]", "_", model_name) + ".onnx"


# avoid loading model from huggingface multiple times, it's time consuming
@functools.lru_cache
def get_tokenizer_and_huggingface_model(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    config = transformers.AutoConfig.from_pretrained(model_name)
    text_one_query = ("Christening a shock-and-awe short-selling outfit requires more creativity. ",)
    if model_name == "xlm-roberta-base":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        onnx_config = transformers.models.xlm_roberta.XLMRobertaOnnxConfig(
            config, "sequence-classification"
        )
        text = text_one_query
    elif model_name == "distilbert-base-uncased":
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.DistilBertForSequenceClassification.from_pretrained(model_name)
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(
            config, "masked-lm"
        )
        text = text_one_query
    elif model_name == "google/mobilebert-uncased":
        model = transformers.MobileBertForNextSentencePrediction.from_pretrained(
            model_name
        )
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(
            config, "masked-lm"
        )
        text = ("where is Jim Henson?", "he is at school from where two blocks away")
        text = ("where is Jim Henson?", "Christening a shock-and-awe short-selling outfit requires more creativity. Hindenburg Research, named after the doomed hydrogen-filled German airship, was founded by Jim Henson in 2017 to hunt for impending corporate disasters, and then hold a torch to them")
    elif model_name == "csarron/mobilebert-uncased-squad-v2":
        model = transformers.MobileBertForQuestionAnswering.from_pretrained(model_name)
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(
            config, "question-answering"
        )
        text = ("Who was Jim Henson?", "Jim Henson was a nice puppet")
    elif model_name == "lordtt13/emo-mobilebert":
        model = transformers.MobileBertForSequenceClassification.from_pretrained(
            model_name
        )
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(
            config, "sequence-classification"
        )
        text = ("Hello, my dog is cute",)
        text = ("Christening a shock-and-awe short-selling outfit requires more creativity. Hindenburg Research, named after the doomed hydrogen-filled German airship, was founded by Jim Henson in 2017 to hunt for impending corporate disasters, and then hold a torch to them",)
    elif model_name == "bert-base-uncased":
        model = transformers.BertForNextSentencePrediction.from_pretrained(
            model_name
        )
        onnx_config = transformers.models.bert.BertOnnxConfig(
            config, "default"
        )
        text = ("Christening a shock-and-awe short-selling outfit requires more creativity. Hindenburg Research, named after the doomed hydrogen-filled German airship, was founded by Jim Henson in 2017 to hunt for impending corporate disasters, and then hold a torch to them",)
    elif model_name == "microsoft/deberta-base":
        model = transformers.DebertaForSequenceClassification.from_pretrained(
            model_name
        )
        onnx_config = transformers.models.deberta.DebertaOnnxConfig(
            config, "sequence-classification"
        )
        text = ("Christening a shock-and-awe short-selling outfit requires more creativity. Hindenburg Research, named after the doomed hydrogen-filled German airship, was founded by Jim Henson in 2017 to hunt for impending corporate disasters, and then hold a torch to them",)
    elif model_name == "gpt2":
        tokenizer = transformers. GPT2Tokenizer.from_pretrained('gpt2')
        model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        #model = transformers.GPT2Model.from_pretrained('gpt2')
        onnx_config = transformers.models.gpt2.GPT2OnnxConfig(
            config, "sequence-classification"
        )
        text = ("Christening a shock-and-awe short-selling outfit requires more creativity. Hindenburg Research, named after the doomed hydrogen-filled German airship, was founded by Jim Henson in 2017 to hunt for impending corporate disasters, and then hold a torch to them",)
    elif model_name == "valhalla/bart-large-sst2":
        model = transformers.BartForSequenceClassification.from_pretrained(
            model_name)
        onnx_config = transformers.models.bart.BartOnnxConfig(
            config, "sequence-classification"
        )
        text = ("Christening a shock-and-awe short-selling outfit requires more creativity. Hindenburg Research, named after the doomed hydrogen-filled German airship, was founded by Jim Henson in 2017 to hunt for impending corporate disasters, and then hold a torch to them",)
    elif model_name == "nghuyong/ernie-1.0-base-zh":
        model = transformers.ErnieForNextSentencePrediction.from_pretrained(
            model_name)
        onnx_config = transformers.models.ernie.ErnieOnnxConfig(
            config, "sequence-classification"
        )
        text = ("Christening a shock-and-awe short-selling outfit requires more creativity. Hindenburg Research, named after the doomed hydrogen-filled German airship, was founded by Jim Henson in 2017 to hunt for impending corporate disasters, and then hold a torch to them",)
    else:
        raise ValueError(f"{model_name} is not supported")
    return tokenizer, model, onnx_config, text


def export_backbone_or_existed(model_name: str):
    """
    To export onnx model from huggingface if user didn't provide one.
    """

    bert_onnx_model = get_this_dir() / get_canonized_name(model_name)
    if bert_onnx_model and bert_onnx_model.exists():
        #print("User provided a customized onnx Model, skip regenerating.")
        return bert_onnx_model

    # fix the seed so we can reproduce the results
    transformers.set_seed(42)
    tokenizer, model, onnx_config, text = get_tokenizer_and_huggingface_model(
        model_name
    )
    inputs = tokenizer(*text, return_tensors="np")

    save_bert_onnx = True
    # tempfile will be removed automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        canonized_name = re.sub(r"[^a-zA-Z0-9]", "_", model_name) + ".onnx"
        onnx_model_path = Path(tmpdir + "/" + canonized_name)
        onnx_inputs, onnx_outputs = transformers.onnx.export(
            tokenizer, model, onnx_config, 16, onnx_model_path
        )
        if save_bert_onnx:
            shutil.copy(onnx_model_path, bert_onnx_model)
        return bert_onnx_model

def get_backbone_onnx_path(model_name: str)->Path:
    bert_onnx_model = export_backbone_or_existed(model_name=model_name)
    return bert_onnx_model

if __name__ == "__main__":
    get_backbone_onnx_path("google/mobilebert-uncased")
