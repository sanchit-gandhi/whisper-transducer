# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Whisper model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/config.json",
}

# fmt: off
NON_SPEECH_TOKENS = [
    1, 2, 7, 8, 9, 10, 14, 25,
    26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
    63, 90, 91, 92, 93, 357, 366, 438, 532, 685,
    705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377,
    1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211,
    4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786,
    11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791,
    17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409,
    34949, 40283, 40493, 40549, 47282, 49146, 50257, 50359, 50360, 50361
]
NON_SPEECH_TOKENS_MULTI = [
    1, 2, 7, 8, 9, 10, 14, 25,
    26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
    63, 90, 91, 92, 93, 359, 503, 522, 542, 873,
    893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627,
    3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647,
    7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793,
    14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675,
    22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865,
    42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362
]
# fmt: on

class WhisperConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`WhisperModel`]. It is used to instantiate a
    Whisper model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Whisper
    [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 51865):
            Vocabulary size of the Whisper model. Defines the number of different tokens that can be represented by the
            `decoder_input_ids` passed when calling [`WhisperModel`]
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel features used per input features. Should correspond to the value used in the
            `WhisperProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_start_token_id (`int`, *optional*, defaults to 50257):
            Corresponds to the "<|startoftranscript|>" token, which is automatically used when no `decoder_input_ids`
            are provided to the `generate` function. It is used to guide the model`s generation process depending on
            the task.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as an encoder/decoder or not.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 256):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to False):
            Scale embeddings by diving by sqrt(d_model).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        max_target_positions (`int`, *optional*, defaults to 448):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        pad_token_id (`int`, *optional*, defaults to 50256):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 50256):
            Begin of stream token id.
        eos_token_id (`int`, *optional*, defaults to 50257):
            End of stream token id.
        suppress_tokens (`List[int]`, *optional*):
            A list containing the non-speech tokens that will be used by the logit processor in the `generate`
            function. NON_SPEECH_TOKENS and NON_SPEECH_TOKENS_MULTI each correspond to the `english-only` and the
            `multilingual` model.
        begin_suppress_tokens (`List[int]`, *optional*, defaults to `[220,50256]`):
            A list containing tokens that will be supressed at the beginning of the sampling process. Initialized as
            the token for `" "` (`blank_token_id`) and the `eos_token_id`
    """
    model_type = "whisper"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=4096,
        num_mel_bins=80,
        encoder_layers=6,
        encoder_attention_heads=4,
        decoder_layers=6,
        decoder_attention_heads=4,
        decoder_ffn_dim=1536,
        encoder_ffn_dim=1536,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        decoder_start_token_id=50257,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=256,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=False,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=1,
        bos_token_id=50257,
        eos_token_id=50256,
        suppress_tokens=None,
        begin_suppress_tokens=[220, 50256],
        blank_token_id=4096,
        token_embedding_dim=512,
        num_lstm_layers=3,
        lstm_hidden_dim=512,
        lstm_layer_norm_epsilon=1e-3,
        lstm_dropout=0.3,
        joiner_activation="relu",
        max_output_length=128,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        # RNN-T
        self.blank_token_id = blank_token_id
        self.token_embedding_dim = token_embedding_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layer_norm_epsilon = lstm_layer_norm_epsilon
        self.lstm_dropout = lstm_dropout

        self.joiner_activation = joiner_activation
        self.max_output_length = max_output_length

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            **kwargs,
        )
