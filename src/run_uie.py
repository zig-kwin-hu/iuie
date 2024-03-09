#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import copy
import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import torch
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint

from peft_moe import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftModel,
    AdaLoraConfig,
    MOELoraConfig,
)
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from model.bloom import BloomForCausalLM_WithLoss
from model.codegen import CodeGenForCausalLM_WithLoss
from model.gpt_neox import GPTNeoXForCausalLM_WithLoss
from model.t5 import MOET5ForConditionalGeneration
from model.t5_utils import MOET5Config
from transformers.models.t5.configuration_t5 import T5Config
# from utils.lorahub import set_lorahub_model

from uie_collator import DataCollatorForUIE
from uie_dataset import gen_cache_path

from uie_trainer import UIETrainer, DenserEvalCallback, SavePeftModelCallback, SaveMetricsCallback, skip_instructions, SaveBestModelsCallback, SkipEpochEvalCallback, EarlyStoppingCallbackWithLog
from compute_metrics import compute_f1, compute_metrics, compute_grouped_metrics

import json
# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def get_best_checkpoints(output_dir):
    paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("best_model_for_")]
    return paths

def print_number_of_trainable_model_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return f"trainable model parameters: {trainable_params}\nall model parameters: {all_param}\npercentage of trainable model parameters: {100 * trainable_params / all_param:.2f}%"
def merge_embeddings(uid2index, embeddings, datasets):
    necessary_index = set()
    for dataset in datasets:
        for example in dataset:
            uid = example['Instance']['unique_id']
            index = uid2index[uid]
            necessary_index.add(index)
    necessary_index = list(necessary_index)
    necessary_index.sort()
    index2newindex = {index: i for i, index in enumerate(necessary_index)}
    new_embeddings = embeddings[necessary_index]
    
    new_uid2index = {}#uid: index2newindex[uid2index[uid]] for uid in uid2index}
    for dataset in datasets:
        for example in dataset:
            uid = example['Instance']['unique_id']
            index = uid2index[uid]
            new_index = index2newindex[index]
            new_uid2index[uid] = new_index
    new_embeddings = torch.nn.Parameter(new_embeddings.clone().detach(), requires_grad=False)
    return new_uid2index, new_embeddings

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={
            "help": "apply LoRA Rank"
        },
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={
            "help": "apply LoRA Alpha"
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "apply LoRA dropout"
        },
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "apply LoRA PEFT training target"
        },
    )
    use_ada_lora: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether use ada lora during training"
        }
    )
    lora_moe_paths: Optional[str] = field(
        default=None,
        metadata={"help": "List of lora adapter paths for LoRA MoE strategy. Only support for inference."}
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "If true, use flash attention 2."},
    )
    moe_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "If true, use moe lora."},
    )
    task_num: Optional[int] = field(
        default=1,
        metadata={"help": "The number of tasks."},
    )
    expert_num: Optional[int] = field(
        default=1,
        metadata={"help": "The number of experts in moe lora."},
    )
    task_embedding_dim: Optional[int] = field(
        default=64,
        metadata={"help": "The embedding dim for task embedding."},
    )
    moe_topk: Optional[int] = field(
        default=-1,
        metadata={"help": "The topk for moe lora."},
    )
    gate_type: Optional[str] = field(
        default="TopKGate",
        metadata={"help": "The gate type for moe lora."},
    )
    gate_loss_type: Optional[str] = field(
        default="no_loss",
        metadata={"help": "The gate loss type for moe lora."},
    )
    add_noise: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add noise to the gate output."},
    )
    regularized: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to regularize the gate output."},
    )
    with_universal: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add universal expert."},
    )
    existing_gate_weight: Optional[str] = field(
        default=None,
        metadata={"help": "The path of existing gate weight."},
    )
    gate_embedding_dim: Optional[int] = field(
        default=None,
        metadata={"help": "The embedding dim for gate embedding."},
    )
    before_moe_lora_gate_embedding_reduction: Optional[int] = field(
        default=-1,
        metadata={"help": "The dim for gate embedding before moe lora gate embedding reduction."},
    )




@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    prompt_file: str = field(
        default=None, metadata={"help": "The prompt file for different preprompt."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_num_instances_per_predict_task: int = field(
        default=None,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    min_negative_labels: Optional[int] = field(
        default=-1,
        metadata={"help": "number of negative label in prompt."}
    )
    min_positive_labels: Optional[int] = field(
        default=-1,
        metadata={"help": "number of positive label in prompt."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    common_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "common dataset name for zero shot."}
    )
    over_sampling: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to over sampling the dataset to max_num_instances_per_task"}
    )#there was a bug here, the type should be bool, but it was str
    ordered_prompt: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to sort prompt options in instruction."}
    )
    embedding_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "which prompt to use to collect the embedding of the input."}
    )



@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    metric_for_best_model: Optional[str] = field(
        default="f1",
        metadata={"help": 'Metric to define best model. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`.'}
    )
    only_save_best_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to only save the best checkpoint."}
    )
    predict_each_dataset_with_best: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to predict each dataset with its own best checkpoint. This will introduce multiple round of prediction."}
    )
    skip_epoch_eval: Optional[int] = field(
        default=0,
        metadata={"help": "Whether to skip epoch eval. If specified, the model will do evaluation every skip_epoch_eval epochs."}
    )
    no_saving: Optional[bool] = field(
        default = False,
        metadata = {"help": "Whether to prevent any saving"}
    )
    test_with_eval: Optional[bool] = field(
        default = False,
        metadata = {"help": "If true, when evaluation, do test as well."}
    )
    use_test_as_eval: Optional[bool] = field(
        default = False,
        metadata = {"help": "If true, when evaluation, do test set as the eval."}
    )
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Define lora training. Auto set to True if lora_target_modules is defined, else False."}
    )
    save_lora_weights_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Save only lora weights on saving checkpoint (without original model, optimizer and trainer states). Auto set to False if lora_target_modules is not defined."}
    )
    embedding_type: Optional[str] = field(
        default=None,
        metadata={"help": "how to collect the embedding of the input."},
    )
    no_resume_training: Optional[bool] = field(
        default=False,
        metadata={"help": "If true, do not resume training from checkpoint."},
    )
    load_in_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "If true, use 4bit training."},
    )
    auto_find_best_lora_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If true, automatically find the best lora checkpoint."},
    )
    early_stopping_patience: Optional[int] = field(
        default=0,
        metadata={"help": "If specified, the model will do early stopping."},
    )
    gate_loss_weight: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The weight for gate loss."},
    ),
    use_sentence_embedding_for_gate: Optional[bool] = field(
        default=False,
        metadata={"help": "If true, use sentence embedding for gate."},
    )
    use_cluster_embedding_for_gate: Optional[bool] = field(
        default=False,
        metadata={"help": "If true, use cluster embedding for gate."},
    )
    cluster_embedding_path: Optional[str] = field( 
        default=None,
        metadata={"help": "The path of cluster embedding."},
    )
    sentence_embedding_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of sentence embedding."},
    )
    cluster_uid2index_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of cluster uid2index."},
    )
    sentence_uid2index_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of sentence uid2index."},
    )
    embedding_sets: Optional[str] = field(
        default='',
        metadata={"help": "The sets to collect the embedding."},
    )
    write_gate_loads: Optional[bool] = field(
        default=False,
        metadata={"help": "Write down the gate loads"},
    )
    max_num_instances_per_predict_task_: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )

    

    


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    assert model_args.moe_lora or not training_args.write_gate_loads, "write_gate_loads only support moe lora"
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            #modified by huzikun, no need to raise error, because there is no checkpoint in the output_dir
            #raise ValueError(
            #    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            #    "Use --overwrite_output_dir to overcome."
            #)
            pass
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)

    training_args.save_lora_weights_only = False if not model_args.lora_target_modules else training_args.save_lora_weights_only
    training_args.use_lora = model_args.lora_target_modules != None
    if model_args.lora_target_modules:
        model_args.lora_target_modules = model_args.lora_target_modules.split(',')
    
    if model_args.lora_moe_paths:
        model_args.lora_moe_paths = model_args.lora_moe_paths.split(',')
    

    # Get the UIE dataset
    print('max num instances per eval task, max num instances per predict task')
    print(data_args.max_num_instances_per_eval_task, data_args.max_num_instances_per_predict_task)
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "uie_dataset.py"),
        data_dir=data_args.data_dir,
        task_config_dir=data_args.task_config_dir,
        instruction_file=data_args.instruction_file,
        instruction_strategy=data_args.instruction_strategy,
        prompt_file=data_args.prompt_file,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        max_num_instances_per_predict_task=data_args.max_num_instances_per_predict_task,
        num_examples=data_args.num_examples,
        over_sampling=data_args.over_sampling,
        min_negative_labels=data_args.min_negative_labels,
        min_positive_labels=data_args.min_positive_labels,
        model_name_or_path=model_args.model_name_or_path,
        model_cache_dir=model_args.cache_dir,
        embedding_prompt=data_args.embedding_prompt,
    )
    raw_datasets.cleanup_cache_files()
    print(data_cache_dir)
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    device_map = None
    bnb_config = None
    if 'bloom' in model_args.model_name_or_path.lower():
        model_class = BloomForCausalLM_WithLoss
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        task_type = TaskType.CAUSAL_LM
    elif 'codegen' in model_args.model_name_or_path.lower():
        model_class = CodeGenForCausalLM_WithLoss
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        task_type = TaskType.CAUSAL_LM
    elif 'neox' in model_args.model_name_or_path.lower():  # add neox
        model_class = GPTNeoXForCausalLM_WithLoss
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        task_type = TaskType.CAUSAL_LM
    elif 'llama' in model_args.model_name_or_path.lower():
        model_class = AutoModelForCausalLM
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
        task_type = TaskType.CAUSAL_LM
        if not training_args.deepspeed:
            device_map = "auto"
        if training_args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
    elif any([k in model_args.model_name_or_path.lower() for k in ['t5', 'instructuie']]) and model_args.moe_lora:
        model_class = MOET5ForConditionalGeneration
        task_type = TaskType.SEQ_2_SEQ_LM
        if not training_args.deepspeed:
            device_map = "auto"
        assert isinstance(config, T5Config)
        config = MOET5Config.from_t5_config(config, model_args.before_moe_lora_gate_embedding_reduction)
    else:
        model_class = AutoModelForSeq2SeqLM
        task_type = TaskType.SEQ_2_SEQ_LM
        if not training_args.deepspeed:
            device_map = "auto"
    kwargs = {
        "pretrained_model_name_or_path": model_args.model_name_or_path,
        "from_tf": bool(".ckpt" in model_args.model_name_or_path),
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "device_map": device_map,
        "torch_dtype": torch.bfloat16 if training_args.bf16 else None,
        "quantization_config": bnb_config,
    }
    if 'llama' in model_args.model_name_or_path.lower():
        kwargs["use_flash_attention_2"] = model_args.use_flash_attention_2
    model = model_class.from_pretrained(**kwargs)
    model.resize_token_embeddings(len(tokenizer))
    if training_args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.use_lora or model_args.use_ada_lora:
        if model_args.moe_lora:
            existing_gate_weight, gate_embedding_dim = None, None
            if type(model_args.existing_gate_weight) == str and os.path.exists(model_args.existing_gate_weight):
                existing_gate_weight = torch.tensor(np.load(model_args.existing_gate_weight))
                gate_embedding_dim = existing_gate_weight.shape[1]
            else:
                gate_embedding_dim = model_args.gate_embedding_dim
            config = MOELoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                task_type=task_type,
                expert_num=model_args.expert_num,
                task_num=model_args.task_num,
                task_embedding_dim=model_args.task_embedding_dim,
                moe_topk=model_args.moe_topk,
                gate_type=model_args.gate_type,
                gate_loss_type=model_args.gate_loss_type,
                add_noise=model_args.add_noise,
                regularized=model_args.regularized,
                with_universal=model_args.with_universal,
                existing_gate_weight=existing_gate_weight,
                gate_embedding_dim=gate_embedding_dim,
            )
        elif not model_args.use_ada_lora:
            config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                task_type=task_type,
            )
        else:
            config = AdaLoraConfig(
                init_r=12,
                target_r=8,
                beta1=0.85,
                beta2=0.85,
                tinit=200,
                tfinal=1000,
                deltaT=10,
                lora_alpha=32,
                lora_dropout=0.1,
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
            )  
        model = get_peft_model(model, config)

        #if model_args.use_ada_lora:
        #    max_memory = get_balanced_memory(
        #        model,
        #        max_memory=None,
        #        no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
        #        dtype='float16',
        #        low_zero=False,
        #    )
#
        #    device_map = infer_auto_device_map(
        #        model,
        #        max_memory=max_memory,
        #        no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
        #        dtype='float16'
        #    )
#
        #    model = dispatch_model(model, device_map=device_map)

        print(print_number_of_trainable_model_parameters(model))
        model.print_trainable_parameters()
        logger.info("Applying LORA peft model")
        
        if training_args.resume_from_checkpoint:
            checkpoint_name = os.path.join(
                training_args.resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    training_args.resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                training_args.resume_from_checkpoint = None
                if os.path.exists(checkpoint_name):
                    print(f"Restarting from LoRA Adapter {checkpoint_name}")
                    adapters_weights = torch.load(checkpoint_name, map_location="cuda")
                    set_peft_model_state_dict(model, adapters_weights)
                else:
                    print(f"LoRA Checkpoint {checkpoint_name} not found")
        elif training_args.auto_find_best_lora_checkpoint:
            fin_path = os.path.join(training_args.output_dir, "eval_metrics_each_epoch_use_test_as_eval_{}.jsonl".format(data_args.max_num_instances_per_predict_task))
            if not os.path.exists(fin_path):
                fin_path = os.path.join(training_args.output_dir, f"eval_metrics_each_epoch_use_test_as_eval.jsonl")
            with open(fin_path, "r") as fin:
                lines = fin.readlines()
                js = [json.loads(line) for line in lines]
                best_f1 = 0
                best_global_step = None
                for j in js:
                    if j["eval"]["eval_f1"] > best_f1:
                        best_f1 = j["eval"]["eval_f1"]
                        best_global_step = j["eval"]["eval_global_step"]
                if best_global_step is not None:
                    checkpoint_name_ = os.path.join(training_args.output_dir, f"checkpoint-{best_global_step}")
                    checkpoint_name = os.path.join(checkpoint_name_, "pytorch_model.bin")
                    if not os.path.exists(checkpoint_name):
                        checkpoint_name = os.path.join(
                            checkpoint_name_, "adapter_model.bin"
                        )  # only LoRA model - LoRA config above has to fit
                        if os.path.exists(checkpoint_name):
                            print(f"Restarting from LoRA Adapter {checkpoint_name}")
                            adapters_weights = torch.load(checkpoint_name, map_location="cuda")
                            set_peft_model_state_dict(model, adapters_weights)
                        else:
                            print(f"LoRA Checkpoint {checkpoint_name} not found")

        model.is_parallelizable = True
        model.model_parallel = True
        model.print_trainable_parameters()
    # elif model_args.lora_moe_paths and not training_args.do_train:
    #     model = set_lorahub_model(
    #         model_args.lora_moe_paths,
    #         checkpoint,
    #         weight_strategy='random',
    #         min_weight_score=-1.5, max_weight_score=1.5
    #     )
    topredict_sets = ['test', 'eval'] if training_args.test_with_eval else ['test']
    if training_args.embedding_sets is not None and len(training_args.embedding_sets) > 0:
        topredict_sets = training_args.embedding_sets.split(',')
    if training_args.do_train or 'train' in topredict_sets:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval or 'eval' in topredict_sets:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    else:
        training_args.metric_for_best_model = None

    if training_args.do_predict or 'test' in topredict_sets:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
    if training_args.use_test_as_eval:
        eval_dataset = copy.deepcopy(predict_dataset)
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    uid2sentid=None
    uid2clusterid=None
    sentence_embeddings_for_gate = None
    cluster_embeddings_for_gate  = None
    if training_args.use_cluster_embedding_for_gate:
        uid2clusterid = json.load(open(training_args.cluster_uid2index_path))
        cluster_embeddings_for_gate = np.load(training_args.cluster_embedding_path)
        cluster_embeddings_for_gate = torch.nn.Parameter(torch.tensor(cluster_embeddings_for_gate, dtype=torch.float32), requires_grad=False)
        print('before merging embeddings', cluster_embeddings_for_gate.shape, len(uid2clusterid))
        uid2clusterid, cluster_embeddings_for_gate = merge_embeddings(uid2clusterid, cluster_embeddings_for_gate, [train_dataset, eval_dataset, predict_dataset])
        print('after merging embeddings', cluster_embeddings_for_gate.shape, len(uid2clusterid))
    if training_args.use_sentence_embedding_for_gate:
        uid2sentid = json.load(open(training_args.sentence_uid2index_path))
        sentence_embeddings_for_gate = np.load(training_args.sentence_embedding_path)
        sentence_embeddings_for_gate = torch.nn.Parameter(torch.tensor(sentence_embeddings_for_gate, dtype=torch.float32), requires_grad=False)
        uid2sentid, sentence_embeddings_for_gate = merge_embeddings(uid2sentid, sentence_embeddings_for_gate, [train_dataset, eval_dataset, predict_dataset])
    
    data_collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        common_dataset_name=data_args.common_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file,
        return_loss_mask=not 'llama' in model_args.model_name_or_path.lower(),
        uid2sentid=uid2sentid,
        uid2clusterid=uid2clusterid, 
        sentence_embeddings_for_gate = sentence_embeddings_for_gate,
        cluster_embeddings_for_gate = cluster_embeddings_for_gate,
        write_gate_loads=training_args.write_gate_loads,
    )
    
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix='dev'):
        decoded_preds = skip_instructions(model, preds, tokenizer, dataset)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions_{data_args.max_num_instances_per_predict_task}.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    def compute_f1_metrics(dataset, preds, save_prefix='dev'):
        decoded_preds = skip_instructions(model, preds, tokenizer, dataset)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        
        result_per_task = compute_f1(dataset, decoded_preds)
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions_{data_args.max_num_instances_per_predict_task}.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result
        
    
    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    callbacks = []
    if training_args.denser_evaluation:
        callbacks.append(DenserEvalCallback)
    if training_args.do_train:
        callbacks.append(SaveMetricsCallback)
    if training_args.only_save_best_model:
        callbacks.append(SaveBestModelsCallback)
    if training_args.skip_epoch_eval > 0:
        assert training_args.evaluation_strategy == 'epoch', "skip_epoch_eval only works with epoch evaluation strategy"
        assert training_args.save_strategy == 'epoch', "skip_epoch_eval only works with epoch save strategy"
        callbacks.append(SkipEpochEvalCallback)
    if training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallbackWithLog(early_stopping_patience=training_args.early_stopping_patience))
    
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty
    training_args.max_num_instances_per_predict_task = data_args.max_num_instances_per_predict_task
    trainer = UIETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        predict_dataset = predict_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_f1_metrics if training_args.embedding_type is None else None,
        callbacks=callbacks,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # For load_best_model() in transformers trainer, because PyTorch doesn't support to load a quantized model from checkpoint into int8, so we need PyTorch's state_dict.
    # if training_args.use_lora:
    #     old_state_dict = model.state_dict
    #     model.state_dict = (
    #         lambda self, *_, **__: get_peft_model_state_dict(
    #             self, old_state_dict()
    #         )
    #     ).__get__(model, type(model))
    
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        
    all_metrics = {"run_name": training_args.run_name}

    # Training
    # 训练epoch数，按照 num_train_epochs 传入，在trainer中解析
    # TODO, train debug, bloomz, flan-t5
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            # modified by huzikun, after resuming, the epoch number should be the remaining epoch number
            training_steps_per_epoch = int(len(train_dataset) / 
                                       (training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps))
            trained_steps = int(checkpoint.split('-')[-1])
            trained_epoch_num = int(trained_steps / training_steps_per_epoch)
            #trainer.args.num_train_epochs = training_args.num_train_epochs - trained_epoch_num
            #print('left num_train_epochs: ', trainer.args.num_train_epochs)
        print('last_checkpoint: ', last_checkpoint)

        from collections import Counter
        c = Counter(p.dtype for p in model.parameters()).most_common()
        print('model parameters dtype: ', c)
        if not training_args.use_lora and not training_args.no_resume_training:
            print('resume training from checkpoint: ', checkpoint)
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            train_result = trainer.train()
        #if not training_args.no_saving:
            #trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        if not training_args.no_saving:
            trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

    # Evaluation
    results = {}
    # in case the batch is shorter than max length, the output should be padded

    if training_args.do_predict:
        logger.info("*** Prediction ***")
        logger.info("*** Loading CheckPoint ***")
        
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        
        checkpoint = None
        
        # Redundant re-initialize model and trainer if do_train True and not predict_each_dataset_with_best because the model output and trainer is expected checkpoint on default (Best model if load_best_model_at_end True OR last checkpoint when False)
        if training_args.predict_each_dataset_with_best and not training_args.no_saving:
            checkpoints = get_best_checkpoints(training_args.output_dir)
            assert len(checkpoints) > 0, "No best checkpoint found"

            ori_model = None
            for checkpoint in checkpoints:
                print(f"loading checkpoint {checkpoint}")
                if not training_args.use_lora:
                    model = model_class.from_pretrained(checkpoint)
                else:
                    if not ori_model:
                        kwargs = {
                            "pretrained_model_name_or_path": model_args.model_name_or_path,
                            "from_tf": bool(".ckpt" in model_args.model_name_or_path),
                            "config": config,
                            "cache_dir": model_args.cache_dir,
                            "revision": model_args.model_revision,
                            "use_auth_token": True if model_args.use_auth_token else None,
                            "device_map": device_map,
                            "torch_dtype": torch.bfloat16 if training_args.bf16 else None,
                            "quantization_config": bnb_config,
                        }
                        if 'llama' in model_args.model_name_or_path.lower():
                            kwargs["use_flash_attention_2"] = model_args.use_flash_attention_2
                        ori_model = model_class.from_pretrained(**kwargs)
                    model = PeftModel.from_pretrained(
                        ori_model,
                        checkpoint
                    )
                training_args.max_num_instances_per_predict_task = data_args.max_num_instances_per_predict_task
                trainer = UIETrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset if training_args.do_train else None,
                    eval_dataset=eval_dataset if training_args.do_eval else None,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_f1_metrics if training_args.embedding_type is None else None,
                    callbacks=callbacks,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                )
                
                predict_results = trainer.predict(
                    predict_dataset,
                    metric_key_prefix="predict",
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id
                )
                metrics = predict_results.metrics
                max_predict_samples = (
                    data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
                )
                metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
                trainer.log({"checkpoint":checkpoint})
                trainer.log_metrics("predict", metrics)
                trainer.save_metrics("predict", metrics)
                all_metrics.update(metrics)
                if training_args.predict_each_dataset_with_best:
                    print(f"Saving predictions for checkpoint {checkpoint}")
                    path = os.path.join(checkpoint, "all_results_{}.json".format(data_args.max_num_instances_per_predict_task))
                    with open(path, "w") as f:
                        json.dump(metrics, f, indent=4, sort_keys=True)
        else:
            dataset_dict = {'train': train_dataset if 'train' in topredict_sets else None,
                            'eval': eval_dataset if 'eval' in topredict_sets else None,
                            'test': predict_dataset if 'test' in topredict_sets else None,
                            }
            model_name = model_args.model_name_or_path.split('/')[-1]
            if training_args.embedding_type is not None:
                if training_args.use_test_as_eval and 'eval' in topredict_sets:
                    topredict_sets.pop(topredict_sets.index('eval'))
            for topredict_set in topredict_sets:
                predict_results = trainer.predict(
                    dataset_dict[topredict_set],
                    metric_key_prefix=topredict_set,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id
                )
                if training_args.embedding_type is not None:
                    embeddings = predict_results.embeddings
                    np.save(os.path.join(training_args.output_dir, 'embeddings_{}_{}_{}_{}_{}.npy'.format(model_name, data_args.embedding_prompt, training_args.embedding_type, topredict_set, data_args.num_examples)), embeddings)
                    continue
                metrics = predict_results.metrics
                max_predict_samples = (
                    data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
                )
                metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
                trainer.log(metrics)
                trainer.log_metrics(topredict_set, metrics)
                trainer.save_metrics(topredict_set, metrics)
                all_metrics.update(metrics)

    if training_args.do_demo:
        logger.info("Serving the model as a demo...")
        user_input = ''
        while True:
            user_input = input("Please enter your input to the model, or enter 'quit' to exit: ")
            if user_input.lower() == "quit":
                break
            inputs = tokenizer([user_input], return_tensors="pt")
            _, preds, _ = trainer.prediction_step(model, inputs=inputs, prediction_loss_only=False)
            print(f"Model generates: {tokenizer.decode(preds[0], skip_special_tokens=True)}\n\n")

    return results


if __name__ == "__main__":
    main()