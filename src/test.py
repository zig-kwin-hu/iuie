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
    DataCollatorForSeq2Seq)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint

from peft_moe import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
    AdaLoraConfig,
)
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from model.bloom import BloomForCausalLM_WithLoss
from model.codegen import CodeGenForCausalLM_WithLoss
from model.gpt_neox import GPTNeoXForCausalLM_WithLoss
from run_uie import ModelArguments, DataTrainingArguments, UIETrainingArguments
# from utils.lorahub import set_lorahub_model

from uie_collator import DataCollatorForUIE
from uie_dataset import gen_cache_path

from uie_trainer import UIETrainer, DenserEvalCallback, SavePeftModelCallback, SaveMetricsCallback, skip_instructions, SaveBestModelsCallback, SkipEpochEvalCallback
from compute_metrics import compute_f1, compute_metrics, compute_grouped_metrics

# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


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
        num_examples=data_args.num_examples,
        over_sampling=data_args.over_sampling,
        min_negative_labels=data_args.min_negative_labels,
        min_positive_labels=data_args.min_positive_labels
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
        device_map = "auto"
    else:
        model_class = AutoModelForSeq2SeqLM
        task_type = TaskType.SEQ_2_SEQ_LM
        if not training_args.deepspeed:
            device_map = "auto"
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        device_map=device_map
    )
    model.resize_token_embeddings(len(tokenizer))
    
    
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

    encoder_outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
    
if __name__ == "__main__":
    main()