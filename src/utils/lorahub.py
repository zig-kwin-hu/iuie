from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy
import random
import nevergrad as ng
from peft_moe import PeftModel, PeftConfig, set_peft_model_state_dict, get_peft_model_state_dict
from functools import partial
from typing import List, Optional, Union


 
def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets

def default_get_loss(example_dataset, model, batch_size):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    data_batch_size = len(example_dataset) if batch_size is None else min(len(example_dataset), batch_size)
    # use gpu if available
    train_dataloader = DataLoader(
        example_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / len(example_dataset["input"])

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def get_score(weights, model, cache, example_dataset, batch_size, get_loss, get_regular):
    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    # Note: the keys are name of model layers (modules), thus final_state_dict has the exact same lora modules but composed state_dict by weights
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
        
    # minimize the metric
    loss = get_loss(example_dataset, model, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    
    return metric_val

def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict

def load_base_model_and_lora_modules(lora_module_list: List[str], model_name_or_path: Optional[str] = None):
    """load base model and lora modules from huggingface model hub

    Args:
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
        model_name_or_path (Optional[str]): base model name, default is None
    """
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load basic model
    default_peft_model_id = lora_module_list[0]
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path
        
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    # 0 is the default model
    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
        
    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = get_peft_model_state_dict(cur_peft_model)

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
               
    return peft_model, cache

def get_weight_scores(lora_nums, strategy, lower_limit, upper_limit):
    if strategy == 'random':
        weights = [random.uniform(upper_limit, lower_limit) for _ in range(lora_nums)]
    else:
        raise "Weights strategy not supported."
    # get_score_partial = partial(get_score, 
    #                             model=model, 
    #                             cache=cache,
    #                             example_dataset=dataset,
    #                             batch_size=batch_size,
    #                             get_loss=get_loss, 
    #                             get_regular=get_regular)
    # # set up the limit of the weights
    # instrum = ng.p.Array(
    #     init=[0] * number_of_loras,
    #     upper=[1.5] * number_of_loras,
    #     lower=[-1.5] * number_of_loras,
    # )
    # optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    # print("> Begin to perform gradient-free optimization ...")
    # recommendation = optimizer.minimize(get_score_partial, verbosity=1)

    return weights

def set_lorahub_model(lora_module_list: List[str], 
                     model_name_or_path=None,
                     weight_strategy = 'random',
                     min_weight_score = -1.5,
                     max_weight_score = 1.5,
                     seed=42):
    # set seed for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    # load model
    model, cache = load_base_model_and_lora_modules(lora_module_list, model_name_or_path)

    weights = get_weight_scores(number_of_loras, weight_strategy, min_weight_score, max_weight_score)
    final_lora = get_final_weights(weights, lora_module_list, cache)
    # set the final weights
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()

    return model