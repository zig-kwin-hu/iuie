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

from peft_moe.import_utils import is_bnb_4bit_available, is_bnb_available

from .config import MOELoraConfig
from .gptq import MOEQuantLinear
from .layer import MOEConv2d, MOEEmbedding, MOELinear, MOELoraLayer
from .model import MOELoraModel


__all__ = ["MOELoraConfig", "MOEConv2d", "MOEEmbedding", "MOELoraLayer", "MOELinear", "MOELoraModel", "MOEQuantLinear"]


if is_bnb_available():
    from .bnb import MOELinear8bitLt

    __all__ += ["MOELinear8bitLt"]

if is_bnb_4bit_available():
    from .bnb import MOELinear4bit

    __all__ += ["MOELinear4bit"]
