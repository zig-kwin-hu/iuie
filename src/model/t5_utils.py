from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class BaseMoEModelOutputWithPast(BaseModelOutput):
    """
    Args:
        num_dropped_tokens: layer idx to the number of dropped tokens
    """

    #last_hidden_state: torch.FloatTensor = None
    #hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[Tuple[torch.Tensor]] = None
    gate_load: Optional[Tuple[list]] = None
    gate_importance: Optional[Tuple[list]] = None


@dataclass
class MOESeq2SeqLMOutput(Seq2SeqLMOutput):
    #last_hidden_state: torch.FloatTensor = None
    #past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    #decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    #cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    #encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    #encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[Tuple[int]] = None
    gate_load: Optional[Tuple[list[torch.Tensor]]] = None
    gate_importance: Optional[Tuple[list[torch.Tensor]]] = None