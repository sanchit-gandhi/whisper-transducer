# coding=utf-8
# Copyright 2023 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Whisper model."""
from dataclasses import dataclass

import math
import random
from typing import Optional, Tuple, List, Union

import torch
import torch.utils.checkpoint
from torch import nn
from tqdm import tqdm

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import  logging
from configuration_whisper import WhisperConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "WhisperConfig"
_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"


WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/whisper-base",
    # See all Whisper models at https://huggingface.co/models?filter=whisper
]

@dataclass
class WhisperRNNTOutput(BaseModelOutput):
    """
    Class for Whisper RNN-Transducer outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        left_context_states (`list(list(torch.FloatTensor))`, *optional*, returned when `is_streaming=True` is passed.
            Whisper internal state representation required for conditioning on the previously processed audio chunk
            during streaming inference.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    left_context_states: Optional[List[List[torch.Tensor]]] = None

# Copied from transformers.models.whisper.modeling_whisper.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.whisper.modeling_whisper._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.whisper.modeling_whisper._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

# Copied from transformers.models.whisper.modeling_whisper.WhisperPositionalEmbedding
class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0):
        return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[-1]]

# Copied from transformers.models.whisper.modeling_whisper.WhisperAttention
class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.whisper.modeling_whisper.WhisperPreTrainedModel
class WhisperPreTrainedModel(PreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (WhisperEncoder)):
            module.gradient_checkpointing = value

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths


    def _get_time_reduced_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # in inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations make sure that all values before the output lengths indices are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


WHISPER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`WhisperConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

WHISPER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Whisper uses the `decoder_start_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read
            [`modeling_whisper._prepare_decoder_attention_mask`] and modify to your needs. See diagram 1 in [the BART
            paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
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

# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoder
class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class RNNTCustomLSTM(torch.nn.Module):
    r"""Custom long-short-term memory (LSTM) block that applies layer normalization
    to internal nodes.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        layer_norm (bool, optional): if `True`, enables layer normalization. (Default: `False`)
        layer_norm_epsilon (float, optional):  value of epsilon to use in
            layer normalization layers (Default: 1e-5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_norm: bool = False,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.x2g = torch.nn.Linear(input_dim, 4 * hidden_dim, bias=(not layer_norm))
        self.p2g = torch.nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.c_norm = torch.nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon)
        self.g_norm = torch.nn.LayerNorm(4 * hidden_dim, eps=layer_norm_epsilon)

        self.hidden_dim = hidden_dim

    def forward(
        self, input: torch.Tensor, state: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Forward pass.

        input (torch.Tensor): with shape *(T, B, D)*.
        state (List[torch.Tensor] or None): list of tensors representing internal state generated in
            preceding invocation of `forward`.
        """
        if state is None:
            batch_size = input.size(1)
            cell_output = torch.zeros(batch_size, self.hidden_dim, device=input.device, dtype=input.dtype)
            cell_state = torch.zeros(batch_size, self.hidden_dim, device=input.device, dtype=input.dtype)
        else:
            cell_output, cell_state = state

        gated_input = self.x2g(input)
        outputs = []
        for gates in gated_input.unbind(0):
            gates = gates + self.p2g(cell_output)
            gates = self.g_norm(gates)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            cell_gate = cell_gate.tanh()
            output_gate = output_gate.sigmoid()
            cell_state = forget_gate * cell_state + input_gate * cell_gate
            cell_state = self.c_norm(cell_state)
            cell_output = output_gate * cell_state.tanh()
            outputs.append(cell_output)

        output = torch.stack(outputs, dim=0)
        state = [cell_output, cell_state]

        return output, state

class RNNTPredictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Args:
        num_symbols (int): size of target token lexicon.
        d_model (int): feature dimension of each output sequence element.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm_epsilon (float, optional): value of epsilon to use in
            LSTM layer normalization layers. (Default: 1e-5)
        lstm_dropout (float, optional): LSTM dropout probability. (Default: 0.0)
    """

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(config.vocab_size, config.token_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(config.token_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                RNNTCustomLSTM(
                    config.token_embedding_dim if idx == 0 else config.lstm_hidden_dim,
                    config.lstm_hidden_dim,
                    layer_norm=True,
                    layer_norm_epsilon=config.lstm_layer_norm_epsilon,
                )
                for idx in range(config.num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=config.lstm_dropout)
        self.linear = torch.nn.Linear(config.lstm_hidden_dim, config.d_model)
        self.output_layer_norm = torch.nn.LayerNorm(config.d_model)

        self.lstm_dropout = config.lstm_dropout

    def forward(
        self,
        input: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size; U: maximum sequence length in batch; D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape *(B, U)* and each element
                mapping to a target symbol, i.e. in range *[0, num_symbols)*.
            lengths (torch.Tensor): with shape *(B,)* and i-th element representing
                number of valid frames for i-th batch element in `input`.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation of `forward`. (Default: `None`)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape *(B, U, d_model)*
                torch.Tensor
                    output lengths, with shape *(B,)* and i-th element representing number of valid elements for i-th
                    batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors representing internal state generated in current invocation
                    of `forward`.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else state[layer_idx])
            lstm_out = self.dropout(lstm_out)
            state_out.append(lstm_state_out)

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), state_out


class RNNTJoiner(torch.nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.activation = ACT2FN[config.joiner_activation]

    def forward(self, source_encodings, target_encodings):
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        joint_encodings = self.activation(joint_encodings)
        output = self.linear(joint_encodings)
        return output

class WhisperForRNNT(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        self.predictor = RNNTPredictor(config)
        self.joiner = RNNTJoiner(config)

        self.blank_token_id = config.blank_token_id
        self.pad_token_id = config.pad_token_id
        self.max_output_length = config.max_output_length

        # Initialize weights and apply final processing
        self.post_init()

    def _gen_next_token_probs(self, encoder_out, predictor_out):
        joined_out = self.joiner(encoder_out, predictor_out)
        joined_out = torch.nn.functional.log_softmax(joined_out, dim=3)
        return joined_out[:, 0, 0]

    def _greedy_decode(self, encoder_out, encoder_lengths):
        past_rnn_state = None
        batch_size = encoder_out.shape[0]
        device = encoder_out.device

        logits = [[] for _ in range(batch_size)]

        last_label = torch.full([batch_size, 1], fill_value=self.blank_token_id, dtype=torch.long, device=device)
        blank_mask = torch.full([batch_size], fill_value=0, dtype=torch.bool, device=device)

        max_time_len = encoder_out.shape[1]
        for time_idx in range(max_time_len):
            encoder_slice = encoder_out.narrow(dim=1, start=time_idx, length=1)

            not_blank = True
            n_decoded_tokens = 0

            blank_mask = blank_mask.fill_(False)
            time_mask = time_idx >= encoder_lengths
            blank_mask = torch.bitwise_or(blank_mask, time_mask)

            while not_blank and (self.max_output_length is None or n_decoded_tokens < self.max_output_length):
                rnn_out, current_rnn_state = self.predictor(last_label, past_rnn_state)
                next_token_logits = self._gen_next_token_probs(encoder_slice, rnn_out)
                next_token_id = next_token_logits.argmax(-1)

                token_is_blank = next_token_id == self.blank_token_id
                blank_mask = torch.bitwise_or(blank_mask, token_is_blank)

                if blank_mask.all():
                    not_blank = False
                else:
                    blank_indices = []
                    if past_rnn_state is not None:
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                    if past_rnn_state is not None:
                        # 3 LSTM layers
                        for layer_id in range(len(past_rnn_state)):
                            # 2 states
                            for state_id in range(len(past_rnn_state[layer_id])):
                                current_rnn_state[layer_id][state_id][blank_indices, :] = past_rnn_state[layer_id][
                                    state_id
                                ][blank_indices, :]

                    next_token_id[blank_indices] = last_label[blank_indices, 0]

                    last_label = next_token_id.clone().view(-1, 1)
                    past_rnn_state = current_rnn_state

                    next_token_id.masked_fill_(blank_mask, self.blank_token_id)
                    for batch_idx, (token_id, token_logits) in enumerate(zip(next_token_id, next_token_logits)):
                        if time_mask[batch_idx] == 0 and token_id != self.blank_token_id:
                            logits[batch_idx].append(token_logits)

                    n_decoded_tokens += 1
        for batch_idx, seq_logits in enumerate(logits):
            if len(seq_logits) > 0:
                logits[batch_idx] = torch.stack(seq_logits, dim=-2)
            else:
                logits[batch_idx] = torch.tensor([], device=device)

        return logits

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        left_context_states: Optional[List[List[torch.Tensor]]] = None,
        is_streaming: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, WhisperRNNTOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones(input_features.shape[:-1], dtype=torch.long, device=self.device)
        )

        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            #left_context_states=left_context_states,
            #is_streaming=is_streaming,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoded_frames = encoder_outputs[0]

        attention_mask = self._get_time_reduced_attention_mask(input_features.shape[1], attention_mask)

        logits = self._greedy_decode(encoded_frames, attention_mask.sum(-1))
        logits = nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=self.pad_token_id)

        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        # TODO: support RNN-T loss
        loss = None
        if not return_dict:
            output = (logits,) + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return WhisperRNNTOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            left_context_states=left_context_states,
        )