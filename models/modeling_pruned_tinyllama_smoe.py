import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import List, Optional, Tuple, Union
from transformers import Cache, DynamicCache
from transformers import LlamaTokenizer
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings

from models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel, LlamaSdpaAttention, LlamaAttention, LlamaRMSNorm, LlamaFlashAttention2
from lib.eval import eval_ppl
from configurations.configuration_pruned_tinyllama_smoe import PrunedLlamaSMoEConfig 

import os
import json
import pandas as pd

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PrunedLlamaSMoEConfig"


# --------- PRUNED LLAMA SMOE STRUCTURE ---------
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Expert <-> LLaMA MLP
class Expert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Router
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, num_selects, instruction="establish", temperature_coefficient=1.0):
        super(NoisyTopkRouter, self).__init__()
        self.num_selects = num_selects
        self.num_experts = num_experts
        
        self.instruction = instruction
        self.temperature_coefficient = temperature_coefficient

        self.topkroute_linear = nn.Linear(n_embed, num_experts, bias=False)
        self.noise_linear = nn.Linear(n_embed, num_experts, bias=False)
    
    def forward(self, x):
        logits = self.topkroute_linear(x)
        
        if self.training:
            noise_logits = self.noise_linear(x)
            noise = torch.randn_like(logits) * F.softplus(noise_logits) * 0.0005
            noisy_logits = logits + noise
            
            if self.instruction == "train_router":
                t = self.temperature_coefficient
                gating_weights, indices = torch.topk(noisy_logits, self.num_experts)
                gating_weights = F.softmax(gating_weights / t, dim=1, dtype=torch.float).to(x.dtype)

            elif self.instruction == "train_expert":
                gating_weights, indices = torch.topk(noisy_logits, self.num_selects)
                gating_weights = F.softmax(gating_weights, dim=1, dtype=torch.float).to(x.dtype)

            else:
                raise ValueError(f"Unsupported instruction during training: {self.instruction}")

        else:
            noisy_logits = logits
            gating_weights, indices = torch.topk(noisy_logits, self.num_selects)
            gating_weights = F.softmax(gating_weights, dim=1, dtype=torch.float).to(x.dtype)

            # self.save_indices(indices, './outputs/intro_case.json')

        return gating_weights, indices
    
    def save_indices(self, indices, filename):
        indices_list = indices.cpu().numpy().tolist()
        with open(filename, 'a') as f:
            json.dump(indices_list, f)
            f.write('\n')


# SMoE Block
class PrunedLlamaSMoEBlock(nn.Module):
    def __init__(self, config: PrunedLlamaSMoEConfig, instruction="establish", temperature_coefficient=1.0, capacity_factor=1.5):
        super(PrunedLlamaSMoEBlock, self).__init__()
        self.num_experts = config.num_experts
        self.num_selects = config.num_selects

        self.router = NoisyTopkRouter(
            config.hidden_size,
            self.num_experts,
            self.num_selects,
            instruction=instruction,
            temperature_coefficient=temperature_coefficient
        )
        self.experts = nn.ModuleList([Expert(config.hidden_size, config.intermediate_size, config.hidden_act) for _ in range(self.num_experts)])

        self.capacity_factor = capacity_factor

    def forward(self, x):
        flat_x = x.view(-1, x.shape[-1])
        gating_weights, indices = self.router(flat_x)
        results = torch.zeros_like(flat_x)
        
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(indices == i)
            results[batch_idx] += gating_weights[batch_idx, nth_expert, None] * expert(flat_x[batch_idx])
        return results.view_as(x)


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


# Establish the Modified Whole Decoder Layer with SMoE Block Based on LlamaDecoderLayer
class PrunedLlamaSMoEDecoderLayer(nn.Module):
    def __init__(self, config: PrunedLlamaSMoEConfig, layer_idx: int, instruction="establish", temperature_coefficient=1.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.pruned_smoe = PrunedLlamaSMoEBlock(
            config,
            instruction=instruction,
            temperature_coefficient=temperature_coefficient
        )

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.pruned_smoe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class PrunedLlamaSMoEPreTrainedModel(LlamaPreTrainedModel):
    config_class = PrunedLlamaSMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PrunedLlamaSMoEDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, PrunedLlamaSMoEModel):
    #         module.gradient_checkpointing = value


# Stack up
class PrunedLlamaSMoEModel(LlamaModel, PrunedLlamaSMoEPreTrainedModel):
    def __init__(self, config: PrunedLlamaSMoEConfig, instruction="establish", temperature_coefficient=1.0):
        super(PrunedLlamaSMoEModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PrunedLlamaSMoEDecoderLayer(
                config,
                layer_idx,
                instruction=instruction,
                temperature_coefficient=temperature_coefficient
            ) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self):
            return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds
        if self.training:
            hidden_states.requires_grad_()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PrunedLlamaSMoEForCausalLM(LlamaForCausalLM, PrunedLlamaSMoEPreTrainedModel):
    def __init__(self, config: PrunedLlamaSMoEConfig, instruction="establish", temperature_coefficient=1.0):
        super(PrunedLlamaSMoEForCausalLM, self).__init__(config)
        self.model = PrunedLlamaSMoEModel(config, instruction=instruction, temperature_coefficient=temperature_coefficient)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# --------- PRUNED LLAMA SMOE ESTABLISHMENT ---------
def establish_pruned_llama_smoe(folder_path, default_config_path, save_model_path = './pruned_smoe_models/tinyllama_smoe'):

    # pruned_models: Original Pruned Models
    # pruned_smoe_model: Established Pruned LLaMA SMoE Model

    # --------- CONFIG & MODEL LOADING ---------
    with open(default_config_path, 'r') as f:
        default_config_dict = json.load(f)
    default_config = PrunedLlamaSMoEConfig.from_dict(default_config_dict)

    def find_all_folders(folder_path):
        file_paths = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isdir(file_path):
                file_paths.append(file_path)
        return file_paths

    def get_pruned_llm(model, cache_dir="./pruned_models"):
        model = LlamaForCausalLM.from_pretrained(
            model,
            config = default_config,
            torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True,
        )
        return model

    # Load the Original Pruned Models
    print("pruned models loading starts...")
    pruned_models_path = find_all_folders(folder_path)

    # Check the num_experts & number of pruned_models
    num_experts = default_config.num_experts

    if len(pruned_models_path) != num_experts:
        raise ValueError(f"Number of pruned models in folder ({len(pruned_models_path)}) does not match num_experts ({num_experts}) in default_config.")
    else:
        print(f"#### number of pruned models: {len(pruned_models_path)}")

    pruned_models = []
    for model_path in pruned_models_path:
        result = get_pruned_llm(model_path)
        pruned_models.append(result)
        
    print("completes!")
    print("*" * 30)


    # --------- PARAMETER SUBSTITUTION ---------
    print("moe model establishment starts...")
    pruned_smoe_model = PrunedLlamaSMoEForCausalLM(default_config)

    pruned_smoe_model.model.embed_tokens = pruned_models[0].model.embed_tokens
    for i in range(default_config.num_hidden_layers):
        pruned_smoe_model.model.layers[i].self_attn = pruned_models[0].model.layers[i].self_attn
        for j in range(default_config.num_experts):
            pruned_smoe_model.model.layers[i].pruned_smoe.experts[j] = pruned_models[j].model.layers[i].mlp
        pruned_smoe_model.model.layers[i].input_layernorm = pruned_models[0].model.layers[i].input_layernorm
        pruned_smoe_model.model.layers[i].post_attention_layernorm = pruned_models[0].model.layers[i].post_attention_layernorm
    pruned_smoe_model.model.norm = pruned_models[0].model.norm
    pruned_smoe_model.lm_head = pruned_models[0].lm_head

    pruned_smoe_model.eval().half()

    tokenizer = LlamaTokenizer.from_pretrained(pruned_models_path[0], use_fast=False)
    print("completes!")

    print(f"Total Model Parameter: {sum(p.numel() for p in pruned_smoe_model.parameters()) / 1000 ** 3:.2f}B")
    print("*" * 30)


    # --------- PPL EVALUATION ---------
    print("moe model ppl test starts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pruned_smoe_model = nn.DataParallel(pruned_smoe_model)
    pruned_smoe_model = pruned_smoe_model.to(device)
    seqlen = 1024

    perplexities = eval_ppl(pruned_smoe_model, tokenizer, seqlen, device)

    test_datasets = ["WikiText2"]
    for dataset, ppl in zip(test_datasets, perplexities):
        print(f"ppl on {dataset}: {ppl}")
    
    pruned_smoe_model = pruned_smoe_model.module if isinstance(pruned_smoe_model, nn.DataParallel) else pruned_smoe_model
    print("*" * 30)

    print(pruned_smoe_model)
    

    # --------- MODEL SAVING ---------
    print("moe model saving starts...")
    if save_model_path:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        pruned_smoe_model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)
    print("completes!")
    print("*" * 30)