# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft_moe.config import PeftConfig
from peft_moe.utils import PeftType


@dataclass
class MOELoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
            For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set
            to `True`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `lora_alpha`.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Lora layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
    task_num: int = field(default=1, metadata={"help": "The number of tasks."})
    task_embedding_dim: int = field(default=64,
                                    metadata={"help": "The dimension of the task embedding."})
    expert_num: int = field(default=4, metadata={"help": "The number of experts."})
    moe_topk: int = field(default=-1, metadata={"help": "The topk num for moe gate."})
    gate_type: str = field(default='TopKGate', metadata={"help": "The type of gate."})
    gate_loss_type: str = field(default='no_loss', metadata={"help": "The type of gate loss."})
    add_noise: bool = field(default=False, metadata={"help": "Whether to add noise to the gate output."})
    regularized: bool = field(default=False, metadata={"help": "Whether to regularize the gate output."})
    with_universal: bool = field(default=False, metadata={"help": "Whether to add universal expert."})


    def __post_init__(self):
        self.peft_type = PeftType.MOELORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
